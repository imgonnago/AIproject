"""
Actor Model
- Qwen2.5-VL 3B 기반 Actor 모델
- 4bit 양자화 + LoRA 학습
- ActorActionTokenizer 통합 (setup() 사용)
- ZeroMQ 클라이언트 통합 (Planner action token 수신)
- critique 텍스트 생성 + 수정된 action vector 출력

실행 전 필요한 파일:
    assets/openvla_action_embeddings.pt
    projection_layer.py
    actor_action_tokenizer.py
"""

import io
import re
import zmq
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType
)
from qwen_vl_utils import process_vision_info

from projection_layer import Projection
from actor_action_tokenizer import ActorActionTokenizer


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

QWEN_MODEL_PATH    = "Qwen/Qwen2.5-VL-3B-Instruct"
OPENVLA_EMBED_PATH = "assets/openvla_action_embeddings.pt"
OPENVLA_DIM        = 4096   # OpenVLA LLM (LLaMA-2) hidden size
QWEN_DIM           = 2048   # Qwen2.5-VL 3B hidden size
PLANNER_HOST       = "localhost"
PLANNER_PORT       = 5555
OPENVLA_VOCAB_SIZE = 32000


# ─────────────────────────────────────────
# Actor Model
# ─────────────────────────────────────────

class ActorModel(nn.Module):
    """
    비판적 재평가 듀얼 시스템의 Actor 모델.

    __init__ 순서 (순서 매우 중요!):
        1. Processor 로드
        2. Projection Layer 초기화
        3. ActorActionTokenizer 초기화
        4. add_tokenizer_vocab() → tokenizer에 action token 256개 추가
           ※ 모델 로드 전에 tokenizer 설정 필수
        5. Qwen 4bit 양자화 로드
        6. action_tokenizer에 qwen 모델 설정
        7. resize_embeddings() + init_action_embeddings()
        8. LoRA 적용
        9. ZeroMQ 클라이언트 연결

    generate() 흐름:
        이미지 + 텍스트
        → ZeroMQ → Planner action token 수신
        → build_prompt() → Qwen Processor → text/image 임베딩
        → action_tokenizer.forward() → action 임베딩 + concat
        → Qwen LLM + LoRA → critique 텍스트 + action token
        → _parse_and_decode_action() → action vector (로봇 실행)
    """

    def __init__(self):
        super(ActorModel, self).__init__()

        # ── 1. Processor 로드 ────────────────────────
        print("Processor 로드 중...")
        self.processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_PATH,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28
        )
        print(f"tokenizer 원래 크기: {len(self.processor.tokenizer)}")

        # ── 2. Projection Layer 초기화 ───────────────
        # OpenVLA 임베딩(4096) → Qwen 임베딩 공간(2048)
        # LLaVA 기반: LayerNorm + Linear + GELU + Linear 구조
        self.projection = Projection(OPENVLA_DIM, QWEN_DIM).to("cuda")

        # ── 3. ActorActionTokenizer 초기화 ───────────
        # qwen_model은 아직 로드 전이므로 None으로 설정
        # 모델 로드 후 action_tokenizer.qwen_model에 설정
        self.action_tokenizer = ActorActionTokenizer(
            processor=self.processor,
            qwen_model=None,
            projection=self.projection
        )

        # ── 4. tokenizer에 action token 추가 ─────────
        # 모델 로드 전에 반드시 먼저 추가해야
        # from_pretrained 시 vocab size가 자동으로 맞춰짐
        self.action_tokenizer.add_tokenizer_vocab()
        print(f"tokenizer 새 크기: {len(self.processor.tokenizer)}")

        # ── 5. Qwen2.5-VL 4bit 양자화 로드 ──────────
        print("Qwen2.5-VL 3B loading... (4bit quantization)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="cuda",
            attn_implementation="sdpa"
        )
        print("Qwen load complete!")

        # ── 6. action_tokenizer에 qwen 모델 설정 ─────
        self.action_tokenizer.qwen_model = self.qwen

        # ── 7. resize + init ──────────────────────────
        # resize_embeddings(): tokenizer 크기에 맞게 임베딩 테이블 확장
        # init_action_embeddings(): 추가된 256개 행을 OpenVLA 임베딩으로 초기화
        self.action_tokenizer.resize_embeddings()
        openvla_embed_weights = torch.load(OPENVLA_EMBED_PATH, weights_only=False)
        self.action_tokenizer.init_action_embeddings(openvla_embed_weights)
        print("ActorActionTokenizer 초기화 완료!")

        # ── 8. LoRA 적용 ─────────────────────────────
        # 4bit 양자화 모델에 LoRA 적용 전 필수 준비
        self.qwen = prepare_model_for_kbit_training(
            self.qwen,
            use_gradient_checkpointing=True
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters()

        # ── 9. ZeroMQ 클라이언트 연결 ────────────────
        zmq_context = zmq.Context()
        self.planner_socket = zmq_context.socket(zmq.REQ)
        self.planner_socket.connect(f"tcp://{PLANNER_HOST}:{PLANNER_PORT}")
        print(f"Planner server connected: {PLANNER_HOST}:{PLANNER_PORT}")

        print("ActorModel initialization complete!")


    # ─────────────────────────────────────────
    # ZeroMQ: Planner action token 요청
    # ─────────────────────────────────────────

    def get_planner_action_tokens(
        self,
        image: Image.Image,
        instruction: str
    ) -> np.ndarray:
        """
        ZeroMQ로 Planner(OpenVLA) 서버에 이미지 + 텍스트 전송,
        action token ID 7개 수신.

        :param image: PIL Image
        :param instruction: 태스크 명령 텍스트
        :return: shape (7,), dtype int, OpenVLA vocab 기준 action token IDs
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        self.planner_socket.send_pyobj({
            "image":       buffer.getvalue(),
            "instruction": instruction
        })

        response = self.planner_socket.recv_pyobj()

        if response.get("status") == "error":
            print(f"[경고] Planner 오류: {response.get('error')}")

        return response["action_tokens"]


    # ─────────────────────────────────────────
    # 프롬프트 구성
    # ─────────────────────────────────────────

    def build_prompt(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_token_ids: np.ndarray
    ) -> dict:
        """
        Qwen Processor 입력 구성.
        시스템 프롬프트 + 태스크 명령 + Planner action 정보 통합.

        :param image: PIL Image
        :param instruction: 태스크 명령 텍스트
        :param planner_action_token_ids: shape (7,), Planner action token IDs
        :return: Qwen Processor 출력 dict
        """
        action_str = " ".join(
            [f"<action_{i}>" for i in planner_action_token_ids]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a robot action critic. "
                    "You receive an image, a task instruction, and a proposed action "
                    "from a planner. Evaluate whether the proposed action is correct, "
                    "explain why it may need modification, and output a corrected action.\n"
                    "Format:\n"
                    "[CRITIQUE] your reasoning [/CRITIQUE]\n"
                    "[ACTION] corrected action tokens [/ACTION]"
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
                            f"Proposed action: {action_str}\n"
                            "Evaluate and correct the proposed action."
                        )
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt"
        ).to("cuda")

        return inputs


    # ─────────────────────────────────────────
    # Forward (GRPO 학습 시 사용)
    # ─────────────────────────────────────────

    def forward(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_tokens=None
    ) -> torch.Tensor:
        """
        GRPO 학습 루프에서 호출. logits 반환.

        planner_action_tokens를 넘기면 OpenVLA 호출 생략.
        rollout에서 저장한 값을 재사용해서 OpenVLA 중복 호출 방지.

        :param image: PIL Image
        :param instruction: 태스크 명령 텍스트
        :param planner_action_tokens: rollout에서 저장한 Planner token (없으면 ZeroMQ 호출)
        :return: logits shape (batch, seq_len, vocab_size)
        """
        # 1. Planner action token 수신 (없으면 ZeroMQ 호출, 있으면 재사용)
        if planner_action_tokens is None:
            planner_action_tokens = self.get_planner_action_tokens(image, instruction)

        # 2. text/image 임베딩 추출
        inputs = self.build_prompt(image, instruction, planner_action_tokens)
        text_image_embeds = self.qwen.base_model.model.model.language_model.embed_tokens(
            inputs["input_ids"]
        )  # (batch, seq_len, 2048)

        # 3. action_tokenizer.forward() → action 임베딩 + concat
        action_token_tensor = torch.tensor(
            planner_action_tokens, dtype=torch.long
        ).to("cuda")

        combined_embeds = self.action_tokenizer.forward(
            action_token_tensor,
            text_image_embeds
        )  # (batch, seq_len+7, 2048)

        # attention mask 확장
        action_mask = torch.ones(1, 7, dtype=torch.long, device="cuda")
        combined_mask = torch.cat(
            [inputs["attention_mask"], action_mask], dim=1
        )

        # 4. Qwen LLM + LoRA
        outputs = self.qwen(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw")
        )

        return outputs.logits


    # ─────────────────────────────────────────
    # Generate (LIBERO 실행 시 사용)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        instruction: str,
        max_new_tokens: int = 50
    ) -> tuple:
        """
        critique 텍스트 + 수정된 action vector 생성.
        LIBERO 환경에서 매 스텝 호출.

        :param image: PIL Image
        :param instruction: 태스크 명령 텍스트
        :param max_new_tokens: 최대 생성 token 수
        :return: (critique: str, action_vector: np.ndarray shape (7,))
        """
        # 1. Planner action token 수신
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)

        # 2. 입력 구성 + 임베딩
        inputs = self.build_prompt(image, instruction, planner_action_tokens)
        text_image_embeds = self.qwen.base_model.model.model.language_model.embed_tokens(
            inputs["input_ids"]
        )

        # 3. action 임베딩 + concat
        action_token_tensor = torch.tensor(
            planner_action_tokens, dtype=torch.long
        ).to("cuda")

        combined_embeds = self.action_tokenizer.forward(
            action_token_tensor,
            text_image_embeds
        )

        action_mask = torch.ones(1, 7, dtype=torch.long, device="cuda")
        combined_mask = torch.cat(
            [inputs["attention_mask"], action_mask], dim=1
        )

        # 4. Qwen LLM 텍스트 생성
        generated_ids = self.qwen.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

        # 5. 출력 디코딩 + 파싱
        output_text = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=False
        )

        critique          = self._parse_critique(output_text)
        action_vector, action_token_ids = self._parse_and_decode_action(output_text)

        # planner_action_tokens도 반환해서 학습 단계에서 재사용
        return critique, action_vector, action_token_ids, planner_action_tokens


    # ─────────────────────────────────────────
    # 출력 파싱
    # ─────────────────────────────────────────

    def _parse_critique(self, output_text: str) -> str:
        """[CRITIQUE] ~ [/CRITIQUE] 사이 텍스트 추출."""
        try:
            start = output_text.index("[CRITIQUE]") + len("[CRITIQUE]")
            end   = output_text.index("[/CRITIQUE]")
            return output_text[start:end].strip()
        except ValueError:
            return ""

    def _parse_and_decode_action(self, output_text: str):
        """
        [ACTION] ~ [/ACTION] 사이 action token 추출
        → action_tokenizer.decode_token_ids_to_actions() → action vector 복원.

        :return: (action_vector: np.ndarray (7,), action_token_ids: np.ndarray (7,))
        """
        try:
            start = output_text.index("[ACTION]") + len("[ACTION]")
            end   = output_text.index("[/ACTION]")
            action_str = output_text[start:end].strip()

            indices = re.findall(r"<action_(\d+)>", action_str)
            indices = np.array([int(i) for i in indices[:7]])

            qwen_action_start = len(self.processor.tokenizer) - 256
            token_ids = indices + qwen_action_start

            return self.action_tokenizer.decode_token_ids_to_actions(token_ids), token_ids

        except (ValueError, IndexError):
            qwen_action_start = len(self.processor.tokenizer) - 256
            dummy_token_ids   = np.array([qwen_action_start] * 7)
            return np.zeros(7), dummy_token_ids


    # ─────────────────────────────────────────
    # 종료
    # ─────────────────────────────────────────

    def close(self):
        """학습 끝나면 호출. ZeroMQ 연결 종료."""
        self.planner_socket.close()
        print("Planner 서버 연결 종료")


# ─────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────

if __name__ == "__main__":
    actor = ActorModel()

    test_image       = Image.new("RGB", (224, 224), color=(100, 150, 200))
    test_instruction = "pick up the black bowl on the left and place it on the plate"

    print("\n=== inference 테스트 ===")
    critique, action_vector = actor.generate(
        image=test_image,
        instruction=test_instruction
    )

    print(f"[Critique]: {critique}")
    print(f"[Action vector]: {action_vector}")
    print(f"[Shape]: {action_vector.shape}")

    actor.close()