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
    # vision model 경로
    # ─────────────────────────────────────────

    def _get_vision_model(self):
        """
        LoRA 적용 후 vision model 경로 반환.
        pixel_values → vision encoder → image_embeds
        """
        try:
            return self.smol.base_model.model.model.vision_model
        except AttributeError:
            for name, module in self.smol.named_modules():
                if "vision_model" in name and not name.endswith("."):
                    return module
            raise AttributeError("vision_model을 찾을 수 없어요.")


    # ─────────────────────────────────────────
    # 공통: combined_embeds 생성
    # ─────────────────────────────────────────

    def _build_combined_embeds(
        self,
        inputs: dict,
        planner_action_tokens: np.ndarray
    ) -> tuple:
        """
        generate()와 forward() 공통 로직.
        pixel_values + input_ids + action_tokens → combined_embeds + combined_mask
        단계별 GPU 올리고 즉시 해제 → VRAM 누적 방지

        :param inputs: build_prompt() 출력 (CPU 텐서)
        :param planner_action_tokens: OpenVLA action token IDs (7,)
        :return: (combined_embeds, combined_mask, input_ids_saved)
        """
        embed_tokens = self._get_embed_tokens()
        vision_model = self._get_vision_model()

        # pixel_values → vision encoder → image_embeds → 즉시 해제
        # num_frames=1로 제한했으므로 shape: (1, 1, 3, H, W) → (1, 3, H, W)
        pixel_values_gpu = inputs["pixel_values"].to("cuda")
        if pixel_values_gpu.dim() == 5:
            pixel_values_gpu = pixel_values_gpu[:, 0]  # 첫 프레임만 (1, 3, H, W)
        image_embeds = vision_model(pixel_values_gpu).last_hidden_state
        del pixel_values_gpu
        torch.cuda.empty_cache()

        # input_ids → text_embeds → 즉시 해제
        input_ids_gpu   = inputs["input_ids"].to("cuda")
        input_ids_saved = input_ids_gpu
        text_embeds     = embed_tokens(input_ids_gpu)
        del input_ids_gpu
        torch.cuda.empty_cache()

        # attention_mask → GPU
        attention_mask_gpu = inputs["attention_mask"].to("cuda")
        torch.cuda.empty_cache()

        # action_tokens → projection → action_embeds
        action_token_tensor = torch.tensor(
            planner_action_tokens, dtype=torch.long
        ).to("cuda")

        # image + text + action concat
        combined_embeds = self.action_tokenizer.forward(
            action_token_tensor,
            text_embeds,
            image_embeds
        )  # (batch, num_patches+seq_len+7, 960)
        del text_embeds, image_embeds, action_token_tensor
        torch.cuda.empty_cache()

        # attention mask 확장
        num_image_patches = combined_embeds.shape[1] - attention_mask_gpu.shape[1] - 7
        image_mask  = torch.ones(1, num_image_patches, dtype=torch.long, device="cuda")
        action_mask = torch.ones(1, 7,                dtype=torch.long, device="cuda")
        combined_mask = torch.cat([image_mask, attention_mask_gpu, action_mask], dim=1)
        del attention_mask_gpu, image_mask, action_mask
        torch.cuda.empty_cache()

        return combined_embeds, combined_mask, input_ids_saved


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
        planner_action_tokens: np.ndarray,
        cached_inputs: dict
    ) -> tuple:
        """
        GRPO 학습 루프에서 호출. (logits, input_ids) 반환.

        generate()에서 저장한 cached_inputs(CPU) 재사용
        → build_prompt(), get_planner_action_tokens() 재호출 없음

        :param planner_action_tokens: rollout에서 저장한 Planner token
        :param cached_inputs: generate()에서 저장한 processor 출력 (CPU)
        :return: (logits, input_ids_saved)
        """
        combined_embeds, combined_mask, input_ids_saved = self._build_combined_embeds(
            cached_inputs, planner_action_tokens
        )

        outputs = self.smol(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
        )
        del combined_embeds, combined_mask
        torch.cuda.empty_cache()

        return outputs.logits, input_ids_saved


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
        비판적 텍스트 + 수정된 action token 생성.
        inputs(CPU)도 반환 → forward()에서 재사용 (build_prompt 재호출 방지)
        """
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)

        inputs = self.build_prompt(image, instruction, planner_action_tokens)

        combined_embeds, combined_mask, _ = self._build_combined_embeds(
            inputs, planner_action_tokens
        )

        generated_ids = self.smol.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )
        del combined_embeds, combined_mask
        torch.cuda.empty_cache()

        output_text = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=False
        )

        critique = self._parse_critique(output_text)
        action_vector, action_token_ids = self._parse_and_decode_action(output_text)

        # inputs(CPU) 반환 → forward()에서 build_prompt 재호출 방지
        return critique, action_vector, action_token_ids, planner_action_tokens, inputs


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

    def _parse_and_decode_action(self, output_text: str):
        """
        [ACTION] ~ [/ACTION] 사이 action token → action vector 복원.

        :return: (action_vector: np.ndarray (7,), action_token_ids: np.ndarray (7,))
        """
        try:
            start = output_text.index("[ACTION]") + len("[ACTION]")
            end   = output_text.index("[/ACTION]")
            action_str = output_text[start:end].strip()

            indices = re.findall(r"<action_(\d+)>", action_str)
            indices = np.array([int(i) for i in indices[:7]])

            smol_action_start = len(self.processor.tokenizer) - 256
            token_ids = indices + smol_action_start

            return self.action_tokenizer.decode_token_ids_to_actions(token_ids), token_ids

        except (ValueError, IndexError):
            smol_action_start = len(self.processor.tokenizer) - 256
            dummy_token_ids   = np.array([smol_action_start] * 7)
            return np.zeros(7), dummy_token_ids


    # ─────────────────────────────────────────
    # 종료
    # ─────────────────────────────────────────

    def close(self):
        self.planner_socket.close()
        print("Planner 서버 연결 종료")