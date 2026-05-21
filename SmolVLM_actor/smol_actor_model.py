"""
Actor Model (SmolVLM2-500M 버전)
- SmolVLM2-500M-Video-Instruct 기반 Actor 모델
- 4bit 양자화 + LoRA 학습
- ActorActionTokenizer 통합
- ZeroMQ 클라이언트 통합 (Planner action token 수신)

변경사항 (Qwen2.5-VL 3B → SmolVLM2-500M):
    모델:    Qwen2_5_VLForConditionalGeneration → SmolVLMForConditionalGeneration
    hidden:  2048 → 960 (SmolLM2-360M)
    vocab:   151921 → ~49408 (49152 + 256 action token)
    VRAM:    ~9GB → ~3~4GB (학습 시)

실행 전 필요한 파일:
    assets/openvla_action_embeddings.pt
    projection_layer.py
    actor_action_tokenizer.py
"""
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import io
import re
import zmq
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import (
    SmolVLMForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType
)

from .smol_projection_layer import Projection
from .smol_action_tokenizer import ActorActionTokenizer


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

SMOL_MODEL_PATH    = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OPENVLA_EMBED_PATH = "assets/openvla_action_embeddings.pt"
OPENVLA_DIM        = 4096   # OpenVLA LLM hidden size
SMOL_DIM           = 960    # SmolLM2-360M hidden_size
PLANNER_HOST       = "localhost"
PLANNER_PORT       = 5555
OPENVLA_VOCAB_SIZE = 32000


# ─────────────────────────────────────────
# Actor Model
# ─────────────────────────────────────────

class ActorModel(nn.Module):
    """
    비판적 재평가 듀얼 시스템의 Actor 모델 (SmolVLM2-500M 버전).

    __init__ 순서:
        1. Processor 로드
        2. Projection Layer 초기화 (4096 → 960)
        3. ActorActionTokenizer 초기화
        4. add_tokenizer_vocab() ← 모델 로드 전 필수
        5. SmolVLM2 4bit 양자화 로드
        6. smol_model 설정 + resize_embeddings() + init_action_embeddings()
        7. LoRA 적용
        8. ZeroMQ 연결
    """

    def __init__(self):
        super(ActorModel, self).__init__()

        # ── 1. Processor 로드 ────────────────────────
        print("Processor 로드 중...")
        self.processor = AutoProcessor.from_pretrained(SMOL_MODEL_PATH)
        print(f"tokenizer 원래 크기: {len(self.processor.tokenizer)}")

        # ── 2. Projection Layer 초기화 ───────────────
        # OpenVLA(4096) → SmolLM2-360M(960)
        self.projection = Projection(OPENVLA_DIM, SMOL_DIM).to("cuda")

        # ── 3. ActorActionTokenizer 초기화 ───────────
        self.action_tokenizer = ActorActionTokenizer(
            processor=self.processor,
            smol_model=None,  # 모델 로드 후 설정
            projection=self.projection
        )

        # ── 4. tokenizer에 action token 추가 ─────────
        # 반드시 모델 로드 전에 추가
        self.action_tokenizer.add_tokenizer_vocab()
        print(f"tokenizer 새 크기: {len(self.processor.tokenizer)}")

        # ── 5. SmolVLM2 4bit 양자화 로드 ─────────────
        print("SmolVLM2-500M loading... (4bit quantization)")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        self.smol = SmolVLMForConditionalGeneration.from_pretrained(
            SMOL_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="cuda"
        )
        print("SmolVLM2 load complete!")

        # ── 6. action_tokenizer에 모델 설정 + 초기화 ──
        self.action_tokenizer.smol_model = self.smol
        self.action_tokenizer.resize_embeddings()

        openvla_embed_weights = torch.load(OPENVLA_EMBED_PATH, weights_only=False)
        self.action_tokenizer.init_action_embeddings(openvla_embed_weights)
        print("ActorActionTokenizer 초기화 완료!")

        # ── 7. LoRA 적용 ─────────────────────────────
        self.smol = prepare_model_for_kbit_training(
            self.smol,
            use_gradient_checkpointing=True
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.smol = get_peft_model(self.smol, lora_config)
        self.smol.print_trainable_parameters()

        # ── 8. ZeroMQ 클라이언트 연결 ────────────────
        zmq_context = zmq.Context()
        self.planner_socket = zmq_context.socket(zmq.REQ)
        self.planner_socket.connect(f"tcp://{PLANNER_HOST}:{PLANNER_PORT}")
        print(f"Planner server connected: {PLANNER_HOST}:{PLANNER_PORT}")

        print("ActorModel initialization complete!")


    # ─────────────────────────────────────────
    # embed_tokens 경로 (LoRA 적용 후)
    # ─────────────────────────────────────────

    def _get_embed_tokens(self):
        """
        LoRA 적용 후 embed_tokens 경로 반환.
        SmolVLM2 구조: base_model.model.model.language_model.embed_tokens
        (실제 경로는 초기 실행 시 named_modules()로 확인 필요)
        """
        try:
            return self.smol.base_model.model.model.language_model.embed_tokens
        except AttributeError:
            # 경로가 다를 경우 자동 탐색
            for name, module in self.smol.named_modules():
                if name.endswith("embed_tokens") and "visual" not in name:
                    return module
            raise AttributeError("embed_tokens를 찾을 수 없어요. named_modules()로 경로 확인 필요")


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
        SmolVLM2 Processor 입력 구성.
        시스템 프롬프트 + 태스크 명령 + Planner action 정보 통합.
        """
        action_str = " ".join(
            [f"<action_{i}>" for i in planner_action_token_ids]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": (
                            "You are a robot action critic. "
                            "Evaluate the proposed action and output a corrected action.\n"
                            "Format:\n"
                            "[CRITIQUE] your reasoning [/CRITIQUE]\n"
                            "[ACTION] corrected action tokens [/ACTION]\n\n"
                            f"Task: {instruction}\n"
                            f"Proposed action: {action_str}\n"
                            "Evaluate and correct the proposed action."
                        )
                    }
                ]
            }
        ]

        # SmolVLM2 processor 사용
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=[image],
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
        """
        GRPO 학습 루프에서 호출. logits 반환.

        :return: logits shape (batch, seq_len, vocab_size)
        """
        # 1. Planner action token 수신
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)

        # 2. text/image 임베딩
        inputs = self.build_prompt(image, instruction, planner_action_tokens)
        embed_tokens = self._get_embed_tokens()
        text_image_embeds = embed_tokens(inputs["input_ids"])  # (batch, seq_len, 960)

        # 3. action 임베딩 + concat
        action_token_tensor = torch.tensor(
            planner_action_tokens, dtype=torch.long
        ).to("cuda")

        combined_embeds = self.action_tokenizer.forward(
            action_token_tensor,
            text_image_embeds
        )  # (batch, seq_len+7, 960)

        # attention mask 확장
        action_mask = torch.ones(1, 7, dtype=torch.long, device="cuda")
        combined_mask = torch.cat(
            [inputs["attention_mask"], action_mask], dim=1
        )

        # 4. SmolVLM2 LLM + LoRA
        outputs = self.smol(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            pixel_values=inputs.get("pixel_values"),
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
        """
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)

        inputs = self.build_prompt(image, instruction, planner_action_tokens)
        embed_tokens = self._get_embed_tokens()
        text_image_embeds = embed_tokens(inputs["input_ids"])

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

        generated_ids = self.smol.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

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
        try:
            start = output_text.index("[CRITIQUE]") + len("[CRITIQUE]")
            end   = output_text.index("[/CRITIQUE]")
            return output_text[start:end].strip()
        except ValueError:
            return ""

    def _parse_and_decode_action(self, output_text: str) -> np.ndarray:
        """
        [ACTION] ~ [/ACTION] 사이 action token → action vector 복원.

        SmolVLM2 기준:
            <action_N>에서 N 추출
            → smol_action_start + N = SmolVLM2 action token ID
            → decode_token_ids_to_actions()
        """
        try:
            start = output_text.index("[ACTION]") + len("[ACTION]")
            end   = output_text.index("[/ACTION]")
            action_str = output_text[start:end].strip()

            indices = re.findall(r"<action_(\d+)>", action_str)
            indices = np.array([int(i) for i in indices[:7]])

            # SmolVLM2 action token ID 변환
            smol_action_start = len(self.processor.tokenizer) - 256
            token_ids = indices + smol_action_start

            return self.action_tokenizer.decode_token_ids_to_actions(token_ids)

        except (ValueError, IndexError):
            return np.zeros(7)


    # ─────────────────────────────────────────
    # 종료
    # ─────────────────────────────────────────

    def close(self):
        self.planner_socket.close()
        print("Planner 서버 연결 종료")