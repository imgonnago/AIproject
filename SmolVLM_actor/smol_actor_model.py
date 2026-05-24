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
import sys
import warnings
warnings.filterwarnings("ignore")
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
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"]
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
    # 공통: input_ids + pixel_values 준비
    # ─────────────────────────────────────────

    def _prepare_inputs(
        self,
        inputs: dict,
        planner_action_tokens: np.ndarray
    ) -> tuple:
        """
        generate()와 forward() 공통 로직.
        input_ids 뒤에 planner action token 붙이기.
        pixel_values는 SmolVLM 내부에서 알아서 처리.

        :param inputs: build_prompt() 출력 (CPU 텐서)
        :param planner_action_tokens: OpenVLA action token IDs (7,)
        :return: (input_ids_with_action, attention_mask_with_action, pixel_values, input_ids_saved)
        """
        # planner action token → SmolVLM tokenizer action token ID로 변환
        # OpenVLA token IDs → bin_indices(0~255) → SmolVLM action token IDs
        bin_indices = self.action_tokenizer.OPENVLA_VOCAB_SIZE - planner_action_tokens
        bin_indices = np.clip(bin_indices, 0, 255)
        smol_action_start = len(self.processor.tokenizer) - 256
        smol_action_token_ids = bin_indices + smol_action_start  # SmolVLM action token ID

        # action token ID → tensor
        action_ids = torch.tensor(
            smol_action_token_ids, dtype=torch.long
        ).unsqueeze(0).to("cuda")  # (1, 7)

        # input_ids 뒤에 action token 붙이기
        input_ids_gpu   = inputs["input_ids"].to("cuda")
        input_ids_saved = input_ids_gpu  # [ACTION] 위치 찾기용
        input_ids_with_action = torch.cat([input_ids_gpu, action_ids], dim=1)

        # attention mask 확장
        attention_mask_with_action = torch.cat([
            inputs["attention_mask"].to("cuda"),
            torch.ones(1, 7, dtype=torch.long, device="cuda")
        ], dim=1)

        # pixel_values: SmolVLM 내부에서 vision encoder 처리
        pixel_values = inputs["pixel_values"].to("cuda")

        return input_ids_with_action, attention_mask_with_action, pixel_values, input_ids_saved


    # ─────────────────────────────────────────
    # ZeroMQ: Planner action token 요청
    # ─────────────────────────────────────────

    def get_planner_action_tokens(
        self,
        image: Image.Image,
        instruction: str
    ) -> np.ndarray:
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

    def _make_messages(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_token_ids: np.ndarray
    ) -> tuple:
        action_str = " ".join(
            [f"<action_{i}>" for i in planner_action_token_ids]
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are a robot action critic who critically evaluates "
                            "action tokens designed by a planner. "
                            "You must critically think about the proposed action, "
                            "explain in text what is wrong with it and why it needs to be corrected. "
                            "Then, based on your critique, you must correct and reason about the action tokens. "
                            "However, if you conclude that the planner's action is already good, "
                            "there is no need to modify it.\n\n"
                            f"Task: {instruction}\n"
                            f"Proposed action: {action_str}\n\n"
                            "Respond in this exact format:\n"
                            "[CRITIQUE] explain what is wrong with the action and why it needs correction [/CRITIQUE]\n"
                            "[ACTION] <action_N> <action_N> <action_N> <action_N> <action_N> <action_N> <action_N> [/ACTION]"
                        )
                    }
                ]
            }
        ]
        return messages

    def _apply_chat_template(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_token_ids: np.ndarray
    ) -> dict:
        messages = self._make_messages(image, instruction, planner_action_token_ids)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )  
        return inputs


    # ─────────────────────────────────────────
    # Forward (GRPO 학습 시 사용)
    # ─────────────────────────────────────────

    def forward(
        self,
        planner_action_tokens: np.ndarray,
        cached_inputs: dict,
        new_tokens: torch.Tensor  # 🌟 새로 추가됨: Actor가 뱉어낸 전체 토큰 (Critique + Action)
    ) -> tuple:
        """
        GRPO 학습 루프에서 호출. (logits, full_input_ids) 반환.
        """
        input_ids_with_action, attention_mask, pixel_values, input_ids_saved = self._prepare_inputs(cached_inputs, planner_action_tokens)

        #프롬프트(input_ids_with_action) 뒤에 Actor가 생성했던 new_tokens를 이어 붙임
        new_tokens_gpu = new_tokens.unsqueeze(0).to("cuda")
        full_input_ids = torch.cat([input_ids_with_action, new_tokens_gpu], dim=1)

        # Attention Mask 확장 (새로 붙은 토큰들도 Attention 연산에 포함)
        new_mask = torch.ones_like(new_tokens_gpu)
        full_attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        outputs = self.smol(
            input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            pixel_values=pixel_values,
        )
        del input_ids_with_action, attention_mask, pixel_values, new_tokens_gpu
        torch.cuda.empty_cache()

        # 학습 로직의 인덱싱 슬라이싱을 위해 통합된 full_input_ids를 리턴합니다.
        return outputs.logits, full_input_ids


    # ─────────────────────────────────────────
    # Generate (LIBERO 실행 시 사용)
    # ─────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        instruction: str,
        max_new_tokens: int = 80
    ) -> tuple:
        """
        비판적 텍스트 + 수정된 action token 생성.
        """
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)
        cached_inputs = self._apply_chat_template(image, instruction, planner_action_tokens)
        input_ids_with_action, attention_mask, pixel_values, _ = self._prepare_inputs(cached_inputs, planner_action_tokens)
        input_length = input_ids_with_action.shape[1]

        generated_ids = self.smol.generate(
            input_ids=input_ids_with_action,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )
        del input_ids_with_action, attention_mask, pixel_values
        torch.cuda.empty_cache()

        new_tokens = generated_ids[0][input_length:]
        output_text = self.processor.decode(new_tokens, skip_special_tokens=False)
        
        print(f"Generated output (Raw): {output_text}")
        
        critique = self._parse_critique(output_text)
        action_vector, action_token_ids = self._parse_and_decode_action(output_text)

        # 🌟 Forward 함수에서 재사용할 수 있도록 new_tokens.cpu()를 반환에 추가합니다. (총 6개)
        return critique, action_vector, action_token_ids, planner_action_tokens, cached_inputs, new_tokens.cpu()


    # ─────────────────────────────────────────
    # 출력 파싱 (유연한 유효성 정규식 보강)
    # ─────────────────────────────────────────

    def _parse_critique(self, output_text: str) -> str:
        try:
            match = re.search(r"\[CRITIQUE\](.*?)(\[\/CRITIQUE\]|$)", output_text, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else ""
        except Exception:
            return ""

    def _parse_and_decode_action(self, output_text: str):
        try:
            match = re.search(r"\[ACTION\](.*?)(\[\/ACTION\]|$)", output_text, re.DOTALL | re.IGNORECASE)
            if not match:
                raise ValueError("ACTION 태그를 찾을 수 없습니다.")
                
            action_str = match.group(1).strip()

            indices = re.findall(r"<action_(\d+)>", action_str)
            if len(indices) < 7:
                raise IndexError("생성된 액션 토큰의 개수가 7개 미만입니다.")
                
            indices = np.array([int(i) for i in indices[:7]])

            smol_action_start = len(self.processor.tokenizer) - 256
            token_ids = indices + smol_action_start

            return self.action_tokenizer.decode_token_ids_to_actions(token_ids), token_ids

        except (ValueError, IndexError, AttributeError) as e:
            smol_action_start = len(self.processor.tokenizer) - 256
            dummy_token_ids   = np.array([smol_action_start] * 7)
            return np.zeros(7), dummy_token_ids


    # ─────────────────────────────────────────
    # 종료
    # ─────────────────────────────────────────

    def close(self):
        self.planner_socket.close()
        print("Planner 서버 연결 종료")