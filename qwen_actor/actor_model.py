"""
Actor Model
- Qwen2.5-VL 3B 기반 Actor 모델
- 4bit 양자화 + LoRA 학습
- ActorActionTokenizer 통합
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
OPENVLA_DIM        = 4096
QWEN_DIM           = 2048
PLANNER_HOST       = "localhost"
PLANNER_PORT       = 5555
OPENVLA_VOCAB_SIZE = 32000


# ─────────────────────────────────────────
# Actor Model
# ─────────────────────────────────────────

class ActorModel(nn.Module):
    """
    비판적 재평가 듀얼 시스템의 Actor 모델.

    __init__ 순서 (순서 중요!):
        1. Processor 로드
        2. Action token 256개 tokenizer에 추가  ← 모델 로드 전에 먼저!
        3. Qwen 4bit 양자화 로드
        4. 임베딩 테이블 resize
        5. Projection Layer 초기화
        6. ActorActionTokenizer 초기화 + init_action_embeddings()
        7. LoRA 적용
        8. ZeroMQ 클라이언트 연결

    핵심: action token을 모델 로드 전에 tokenizer에 추가해야
          resize_token_embeddings()가 정상 작동함
          (4bit 양자화 모델은 로드 후 resize가 제한됨)
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

        # ── 2. Action token 256개 tokenizer에 추가 ───
        # 반드시 모델 로드 전에 추가해야 resize가 정상 작동함
        print(f"tokenizer 원래 크기: {len(self.processor.tokenizer)}")
        action_tokens = [f"<action_{i}>" for i in range(256)]
        num_added = self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": action_tokens}
        )
        print(f"추가된 token 수: {num_added}")
        print(f"tokenizer 새 크기: {len(self.processor.tokenizer)}")

        # ── 3. Qwen2.5-VL 4bit 양자화 로드 ──────────
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

        # ── 4. 임베딩 테이블 resize ──────────────────
        new_vocab_size = len(self.processor.tokenizer)
        self.qwen.resize_token_embeddings(new_vocab_size)
        print(f"임베딩 테이블 크기: {self.qwen.get_input_embeddings().weight.shape}")

        # ── 5. Projection Layer 초기화 ───────────────
        self.projection = Projection(OPENVLA_DIM, QWEN_DIM).to("cuda")

        # ── 6. ActorActionTokenizer 초기화 ───────────
        # add_tokenizer_vocab, resize_embeddings는 위에서 이미 처리했으므로
        # init_action_embeddings만 호출 (setup() 대신)
        self.action_tokenizer = ActorActionTokenizer(
            processor=self.processor,
            qwen_model=self.qwen,
            projection=self.projection
        )

        openvla_embed_weights = torch.load(OPENVLA_EMBED_PATH)
        self.action_tokenizer.init_action_embeddings(openvla_embed_weights)
        print("ActorActionTokenizer 초기화 완료!")

        # ── 7. LoRA 적용 ─────────────────────────────
        self.qwen = prepare_model_for_kbit_training(self.qwen)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.qwen = get_peft_model(self.qwen, lora_config)
        self.qwen.print_trainable_parameters()

        # ── 8. ZeroMQ 클라이언트 연결 ────────────────
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
        ZeroMQ로 Planner 서버에 이미지 + 텍스트 전송,
        action token ID 7개 수신.
        """
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        self.planner_socket.send_pyobj({
            "image": buffer.getvalue(),
            "instruction": instruction
        })

        response = self.planner_socket.recv_pyobj()

        if response.get("status") == "error":
            print(f"[경고] Planner 오류: {response.get('error')}")

        return response["action_tokens"]  # shape (7,), dtype int


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
        instruction: str
    ) -> torch.Tensor:
        """GRPO 학습 루프에서 호출."""

        # 1. ZeroMQ → Planner action token 수신
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)

        # 2. Qwen Processor → text/image 임베딩
        inputs = self.build_prompt(image, instruction, planner_action_tokens)
        text_image_embeds = self.qwen.base_model.model.model.language_model.embed_tokens(inputs["input_ids"])

        # 3. ActorActionTokenizer.forward() → action 임베딩 + concat
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
        """critique 텍스트 + 수정된 action vector 생성."""

        # 1. ZeroMQ → Planner action token 수신
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)

        # 2. 입력 구성 + 임베딩
        inputs = self.build_prompt(image, instruction, planner_action_tokens)
        text_image_embeds = self.qwen.base_model.model.model.language_model.embed_tokens(inputs["input_ids"])

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

        critique      = self._parse_critique(output_text)
        action_vector = self._parse_and_decode_action(output_text)

        return critique, action_vector


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

    def _parse_and_decode_action(self, output_text: str) -> np.ndarray:
        """[ACTION] ~ [/ACTION] 사이 action token → action vector 복원."""
        try:
            start = output_text.index("[ACTION]") + len("[ACTION]")
            end   = output_text.index("[/ACTION]")
            action_str = output_text[start:end].strip()

            indices   = re.findall(r"<action_(\d+)>", action_str)
            indices   = np.array([int(i) for i in indices[:7]])
            token_ids = OPENVLA_VOCAB_SIZE - indices

            return self.action_tokenizer.decode_token_ids_to_actions(token_ids)

        except (ValueError, IndexError):
            return np.zeros(7)


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