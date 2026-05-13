"""
Actor Model (Qwen2.5-VL 3B 기반)
- Planner(OpenVLA)에서 받은 action token을 비판적으로 재평가
- critique 텍스트 생성 후 수정된 action token 출력
"""

import numpy as np
import torch
import torch.nn as nn
from qwen_actor.projection_layer import ProjectionLayer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import get_peft_model, LoraConfig, TaskType
from qwen_vl_utils import process_vision_info
from PIL import Image


# zeroMQ를 구현하지 않아서 전체 코드는 아직 작성하지 않음. 모든 코드 작성 후 actor_model파일을 작성해야함.


openvla_dim = 4096  # OpenVLA LLaMA-2 hidden size
qwen_dim    = 2048  # Qwen2.5-VL 3B hidden size
# ─────────────────────────────────────────
# 2. Actor Model
# ─────────────────────────────────────────

class ActorModel(nn.Module):
    """
    비판적 재평가 듀얼 시스템의 Actor 모델.

    입력:
        - 이미지 (로봇 카메라)
        - 태스크 명령 텍스트
        - Planner(OpenVLA)가 생성한 action token IDs (7개 정수)

    출력:
        - critique 텍스트 ("~때문에 action을 수정해야 한다")
        - 수정된 action token IDs (7개 정수)

    학습 파라미터:
        - ProjectionLayer (전체)
        - Qwen LLM의 LoRA 모듈만
    """

    def __init__(self):
        super(ActorModel, self).__init__()

        # ── 2-1. Qwen2.5-VL 로드 ──────────────────
        print("Qwen2.5-VL 3B 로드 중...")
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28
        )
        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

        # ── 2-2. Qwen tokenizer에 action token 256개 추가 ──
        # OpenVLA의 256개 bin에 대응하는 special token 추가
        # Qwen이 action token을 출력할 수 있게 vocabulary 확장
        action_tokens = [f"<action_{i}>" for i in range(256)]
        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": action_tokens}
        )
        # vocabulary 크기 변경에 맞춰 임베딩 테이블 크기 조정
        self.qwen.resize_token_embeddings(len(self.processor.tokenizer))

        # action token ID 범위 저장 (나중에 출력 파싱에 사용)
        self.action_token_start_id = (
            self.processor.tokenizer.convert_tokens_to_ids("<action_0>")
        )

        # ── 2-3. Qwen에 LoRA 적용 ──────────────────
        # LLM 전체를 학습하지 않고 LoRA 모듈만 학습
        # 나머지 Qwen 파라미터는 frozen 상태 유지
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,                           # LoRA rank
            lora_alpha=32,                  # LoRA scaling factor
            lora_dropout=0.05,
            target_modules=[                # attention 레이어에만 적용
                "q_proj", "k_proj",
                "v_proj", "o_proj"
            ]
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters()  # 학습 파라미터 수 출력

        # ── 2-4. OpenVLA 임베딩 테이블 로드 (frozen) ──
        # openvla_inference.py의 --extract 옵션으로 미리 추출해둔 파일
        # action token ID → 4096차원 벡터 변환에 사용
        action_embed_weights = torch.load(
            "assets/openvla_action_embeddings.pt",
            map_location="cpu"
        )  # shape: (256, 4096)
        self.openvla_embedding = nn.Embedding.from_pretrained(
            action_embed_weights,
            freeze=True  # 학습 안됨 (OpenVLA가 학습한 의미 보존)
        )

        # bin_centers: action token ID → 연속값 복원에 사용 (Detokenizer)
        self.bin_centers = np.load("assets/openvla_bin_centers.npy")

        # ── 2-5. Projection Layer 초기화 ──────────────
        # 학습 대상: OpenVLA 임베딩 공간을 Qwen 공간으로 정렬
        self.projection = ProjectionLayer(openvla_dim, qwen_dim)

        print("Actor 모델 초기화 완료!")


    # ─────────────────────────────────────────
    # 3. action token 임베딩 변환
    # ─────────────────────────────────────────

    def encode_action_tokens(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Planner에서 받은 action token ID를 Qwen LLM이 이해할 수 있는
        임베딩 벡터로 변환한다.

        흐름:
            action token IDs (정수 7개)
            → OpenVLA 임베딩 테이블 (frozen) → (7, 4096)
            → Projection Layer (학습됨)       → (7, 2048)

        Args:
            action_token_ids: shape (batch, 7), dtype int
                Planner(OpenVLA)가 생성한 action token ID

        Returns:
            shape (batch, 7, 2048) — Qwen 임베딩 공간의 action 벡터
        """
        # OpenVLA 임베딩 테이블로 4096차원 변환
        action_embeds = self.openvla_embedding(action_token_ids)  # (batch, 7, 4096)

        # Projection Layer로 Qwen 공간(2048차원)으로 변환
        action_embeds = self.projection(action_embeds)            # (batch, 7, 2048)

        return action_embeds


    # ─────────────────────────────────────────
    # 4. 프롬프트 구성
    # ─────────────────────────────────────────

    def build_prompt(
        self,
        image: Image.Image,
        instruction: str,
        action_token_ids: np.ndarray
    ) -> dict:
        """
        Qwen Processor에 넣을 입력을 구성한다.
        이미지 + 태스크 명령 + action token 정보를 하나의 프롬프트로 통합.

        Args:
            image: PIL Image (로봇 카메라 이미지)
            instruction: 태스크 명령 텍스트
            action_token_ids: Planner가 생성한 action token IDs (7개)

        Returns:
            Qwen Processor 출력 (input_ids, attention_mask, pixel_values 등)
        """
        # action token을 텍스트로 표현 (프롬프트 안에 위치 표시용)
        action_token_str = " ".join(
            [f"<action_{i}>" for i in action_token_ids]
        )

        # 시스템 프롬프트 + 태스크 명령 + action 정보 통합
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot action critic. "
                    "You receive an image, a task instruction, and a proposed action "
                    "from a planner. Evaluate whether the proposed action is correct, "
                    "explain why it may need modification, and output a corrected action. "
                    "Format your response as:\n"
                    "[CRITIQUE] your reasoning here [/CRITIQUE]\n"
                    "[ACTION] corrected action tokens here [/ACTION]"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            f"Task: {instruction}\n"
                            f"Proposed action: {action_token_str}\n"
                            "Evaluate and correct the proposed action."
                        )
                    }
                ]
            }
        ]

        # Qwen chat template 적용
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 이미지 전처리
        image_inputs, _ = process_vision_info(messages)

        # Processor로 최종 입력 구성
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt"
        ).to("cuda")

        return inputs


    # ─────────────────────────────────────────
    # 5. Forward (학습 시 사용)
    # ─────────────────────────────────────────

    def forward(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_token_ids: np.ndarray
    ) -> torch.Tensor:
        """
        Actor의 forward pass. GRPO 학습 루프에서 호출됨.

        흐름:
            1. 이미지 + 텍스트 → Qwen Processor → text/image 임베딩
            2. action token IDs → OpenVLA 임베딩 → Projection → action 임베딩
            3. 두 임베딩 concat → Qwen LLM + LoRA
            4. logits 반환 (GRPO loss 계산에 사용)

        Args:
            image: PIL Image
            instruction: 태스크 명령 텍스트
            planner_action_token_ids: np.ndarray, shape (7,)

        Returns:
            logits: shape (batch, seq_len, vocab_size)
        """

        # ── Step 1: 이미지 + 텍스트 입력 구성 ──
        inputs = self.build_prompt(image, instruction, planner_action_token_ids)

        # ── Step 2: 텍스트/이미지 임베딩 추출 ──
        # Qwen 내부 임베딩 테이블로 input_ids → 임베딩 변환
        text_image_embeds = self.qwen.model.embed_tokens(
            inputs["input_ids"]
        )  # (batch, seq_len, 2048)

        # ── Step 3: action 임베딩 변환 ──
        action_ids = torch.tensor(
            planner_action_token_ids, dtype=torch.long
        ).unsqueeze(0).to("cuda")  # (1, 7)

        action_embeds = self.encode_action_tokens(action_ids)  # (1, 7, 2048)

        # ── Step 4: 이미지/텍스트 임베딩 + action 임베딩 concat ──
        # action 임베딩을 텍스트 임베딩 뒤에 붙임
        combined_embeds = torch.cat(
            [text_image_embeds, action_embeds], dim=1
        )  # (batch, seq_len+7, 2048)

        # attention mask도 action 길이만큼 확장
        action_mask = torch.ones(
            1, 7, dtype=torch.long, device="cuda"
        )
        combined_mask = torch.cat(
            [inputs["attention_mask"], action_mask], dim=1
        )

        # ── Step 5: Qwen LLM + LoRA 통과 ──
        outputs = self.qwen(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw")
        )

        return outputs.logits


    # ─────────────────────────────────────────
    # 6. Inference (평가 시 사용)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_token_ids: np.ndarray,
        max_new_tokens: int = 100
    ) -> tuple[str, np.ndarray]:
        """
        critique 텍스트와 수정된 action vector를 생성한다.
        평가(inference) 시 사용. gradient 계산 없음.

        Args:
            image: PIL Image
            instruction: 태스크 명령 텍스트
            planner_action_token_ids: Planner의 action token IDs (7개)
            max_new_tokens: 최대 생성 token 수

        Returns:
            critique: str — Actor가 생성한 비판 텍스트
            corrected_action: np.ndarray, shape (7,) — 수정된 action vector
        """

        # 입력 구성
        inputs = self.build_prompt(image, instruction, planner_action_token_ids)

        # action 임베딩 변환 및 concat
        action_ids = torch.tensor(
            planner_action_token_ids, dtype=torch.long
        ).unsqueeze(0).to("cuda")
        action_embeds = self.encode_action_tokens(action_ids)

        text_image_embeds = self.qwen.model.embed_tokens(inputs["input_ids"])
        combined_embeds = torch.cat([text_image_embeds, action_embeds], dim=1)

        action_mask = torch.ones(1, 7, dtype=torch.long, device="cuda")
        combined_mask = torch.cat([inputs["attention_mask"], action_mask], dim=1)

        # 텍스트 생성
        generated_ids = self.qwen.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        # 출력 디코딩
        output_text = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=False
        )

        # critique 텍스트 파싱
        critique = self._parse_critique(output_text)

        # action token 파싱 → action vector 복원
        corrected_action = self._parse_and_decode_action(output_text)

        return critique, corrected_action


    # ─────────────────────────────────────────
    # 7. 출력 파싱 함수들
    # ─────────────────────────────────────────

    def _parse_critique(self, output_text: str) -> str:
        """
        LLM 출력에서 [CRITIQUE] ~ [/CRITIQUE] 사이의 텍스트를 추출한다.

        Args:
            output_text: Qwen LLM 전체 출력 텍스트

        Returns:
            critique 텍스트 (없으면 빈 문자열)
        """
        try:
            start = output_text.index("[CRITIQUE]") + len("[CRITIQUE]")
            end   = output_text.index("[/CRITIQUE]")
            return output_text[start:end].strip()
        except ValueError:
            return ""


    def _parse_and_decode_action(self, output_text: str) -> np.ndarray:
        """
        LLM 출력에서 [ACTION] ~ [/ACTION] 사이의 action token을 추출하고
        Action Detokenizer로 연속값 action vector로 복원한다.

        흐름:
            [ACTION] <action_127> <action_203> ... [/ACTION]
            → token ID 추출 (7개)
            → bin_centers로 연속값 복원
            → shape (7,) float array

        Args:
            output_text: Qwen LLM 전체 출력 텍스트

        Returns:
            action vector: np.ndarray, shape (7,), float
                (x, y, z, roll, pitch, yaw, gripper)
        """
        try:
            start = output_text.index("[ACTION]") + len("[ACTION]")
            end   = output_text.index("[/ACTION]")
            action_str = output_text[start:end].strip()

            # <action_N> 형태에서 N 추출
            import re
            action_indices = re.findall(r"<action_(\d+)>", action_str)
            action_indices = [int(i) for i in action_indices[:7]]

            # bin_centers로 연속값 복원 (Detokenizer)
            action_indices = np.clip(action_indices, 0, len(self.bin_centers) - 1)
            return self.bin_centers[action_indices]

        except (ValueError, IndexError):
            # 파싱 실패 시 zero action 반환
            return np.zeros(7)


# ─────────────────────────────────────────
# 8. 테스트
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Actor 모델 초기화
    actor = ActorModel()

    # 테스트용 더미 입력
    test_image        = Image.new("RGB", (224, 224), color=(100, 150, 200))
    test_instruction  = "pick up the black bowl on the left and place it on the plate"
    test_action_ids   = np.array([127, 203, 89, 156, 201, 88, 134])  # Planner 출력

    print("\n=== inference 테스트 ===")
    critique, corrected_action = actor.generate(
        image=test_image,
        instruction=test_instruction,
        planner_action_token_ids=test_action_ids
    )

    print(f"[Planner action token IDs]: {test_action_ids}")
    print(f"[Critique]: {critique}")
    print(f"[Corrected action vector]: {corrected_action}")
    print(f"[Shape]: {corrected_action.shape}")  # (7,)