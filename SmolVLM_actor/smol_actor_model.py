"""
Actor Model (SmolVLM2-500M, 출력 형식 수정 버전)

[핵심 수정]
문제: SmolVLM2는 <action_N> 특수 토큰 생성법을 모름
      → SFT 없이 GRPO만으로 cold start 불가
      → 파싱 실패 → zeros → 보상 없음 → 학습 불가

해결:
    ① 출력 형식: <action_N> 토큰 → 정수 (0~255)
       모델이 자연스럽게 생성 가능한 형식으로 변경
    ② 프롬프트: → <action_N> 입력 제거 (모델이 따라씀)
       연속값만 표시 + 명확한 예시 추가
    ③ 파서: lenient (부분 파싱 허용)
       파싱 실패 시 zeros → 플래너 값 fallback
       (최소한 플래너 수준의 행동 보장 → 보상 신호 유지)
    ④ max_new_tokens 증가 (70 → 100/150)
       응답이 잘려서 ACTION 태그 미생성 방지
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
# VRAM 자동 설정
# ─────────────────────────────────────────

def get_vram_config() -> dict:
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    name     = torch.cuda.get_device_properties(0).name
    print(f"[VRAM 감지] GPU: {name}, VRAM: {total_gb:.1f} GB")

    if total_gb >= 10:
        cfg = dict(group_size=3, max_new_tokens=50, lora_r=8,  lora_alpha=16, use_8bit_adam=False, label="12GB")
    elif total_gb >= 7:
        cfg = dict(group_size=3, max_new_tokens=40, lora_r=4,  lora_alpha=8,  use_8bit_adam=True,  label="8GB")
    else:
        cfg = dict(group_size=1, max_new_tokens=30,  lora_r=4,  lora_alpha=8,  use_8bit_adam=True,  label="6GB")

    print(f"[VRAM 설정] {cfg['label']}: group={cfg['group_size']}, "
          f"max_new={cfg['max_new_tokens']}, lora_r={cfg['lora_r']}")
    return cfg


SMOL_MODEL_PATH    = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
OPENVLA_EMBED_PATH = "assets/openvla_action_embeddings.pt"
OPENVLA_DIM        = 4096
SMOL_DIM           = 960
PLANNER_HOST       = "localhost"
PLANNER_PORT       = 5555
OPENVLA_VOCAB_SIZE = 32000

ACTION_DIM_NAMES = ["x_move", "y_move", "z_move", "roll", "pitch", "yaw", "gripper"]


# ─────────────────────────────────────────
# Actor Model
# ─────────────────────────────────────────

class ActorModel(nn.Module):

    def __init__(self):
        super(ActorModel, self).__init__()
        self.vram_cfg = get_vram_config()

        # ── 1. Processor ─────────────────────────────
        print("Processor 로드 중...")
        self.processor = AutoProcessor.from_pretrained(SMOL_MODEL_PATH)

        # ── 2. Projection ────────────────────────────
        self.projection = Projection(OPENVLA_DIM, SMOL_DIM).to("cuda")

        # ── 3. ActionTokenizer ───────────────────────
        self.action_tokenizer = ActorActionTokenizer(
            processor=self.processor,
            smol_model=None,
            projection=self.projection
        )
        self.action_tokenizer.add_tokenizer_vocab()

        # ── 4. SmolVLM2 4bit 로드 ───────────────────
        print("SmolVLM2-500M loading...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        self.smol = SmolVLMForConditionalGeneration.from_pretrained(
            SMOL_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="cuda",
            attn_implementation="eager"
        )

        # ── 5. Action embedding 초기화 ───────────────
        self.action_tokenizer.smol_model = self.smol
        self.action_tokenizer.resize_embeddings()
        openvla_embed_weights = torch.load(OPENVLA_EMBED_PATH, weights_only=False)
        self.action_tokenizer.init_action_embeddings(openvla_embed_weights)

        # ── 6. LoRA + sparse embedding hook ─────────
        self.smol = prepare_model_for_kbit_training(self.smol, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.vram_cfg["lora_r"],
            lora_alpha=self.vram_cfg["lora_alpha"],
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj"]
        )
        self.smol = get_peft_model(self.smol, lora_config)
        self.smol.print_trainable_parameters()
        self._register_action_only_embedding_hook()

        # ── 7. ZeroMQ ────────────────────────────────
        zmq_context = zmq.Context()
        self.planner_socket = zmq_context.socket(zmq.REQ)
        self.planner_socket.connect(f"tcp://{PLANNER_HOST}:{PLANNER_PORT}")
        print("ActorModel initialization complete!")

    def _register_action_only_embedding_hook(self):
        smol_action_start = len(self.processor.tokenizer) - 256
        embed_layer = self.smol.get_input_embeddings()
        embed_layer.weight.requires_grad_(True)

        def _sparse_hook(grad):
            sparse = torch.zeros_like(grad)
            sparse[smol_action_start:] = grad[smol_action_start:]
            return sparse

        embed_layer.weight.register_hook(_sparse_hook)
        print(f"[sparse hook] action embedding 행({smol_action_start}~)만 학습")


    # ─────────────────────────────────────────
    # ZeroMQ
    # ─────────────────────────────────────────

    def get_planner_action_tokens(self, image: Image.Image, instruction: str) -> np.ndarray:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.planner_socket.send_pyobj({"image": buffer.getvalue(), "instruction": instruction})
        response = self.planner_socket.recv_pyobj()
        if response.get("status") == "error":
            print(f"[Planner 오류] {response.get('error')}")
        return response["action_tokens"]


    # ─────────────────────────────────────────
    # 프롬프트 구성
    # ─────────────────────────────────────────

    def _make_messages(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_token_ids: np.ndarray
    ) -> list:
        """
        [수정] 출력 형식: <action_N> 토큰 → 정수 (0~255)

        수정 전 문제:
            입력에 '→ <action_128>' 표시 → 모델이 이 형식을 출력에 복사
            [ACTION] <action_N> 요구 → 모델이 템플릿 텍스트 그대로 출력

        수정 후:
            입력: 연속값만 표시 (→ <action_N> 제거)
            출력: 7개 정수 (0~255) → 모델이 자연스럽게 생성 가능
            예시: 명확한 입출력 예시 추가 → 형식 학습 가속
        """
        bin_indices       = self.action_tokenizer.openvla_ids_to_bin_indices(planner_action_token_ids)
        continuous_values = self.action_tokenizer.bin_indices_to_continuous(bin_indices)

        # 연속값만 표시 (→ <action_N> 제거)
        action_lines = "\n".join([
            f"  {dim}: {val:+.3f}"
            for dim, val in zip(ACTION_DIM_NAMES, continuous_values)
        ])

        # 플래너 bin 값 (예시 fallback 표시용)
        planner_str = " ".join(str(b) for b in bin_indices)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Task: {instruction}\n\n"
                    f"Proposed robot arm action (7-DOF):\n{action_lines}\n\n"
                    "Values are in range [-1.0, +1.0]. Bin index 0=minimum(-1.0), "
                    "128=neutral(0.0), 255=maximum(+1.0).\n\n"
                    "You are a robot action critic.\n"
                    "You have to evaluate the proposed action and correct it if necessary.\n"
                    "Look at the image. Evaluate the action for this task.\n"
                    "If correct, keep the same bin values. If not, explain why it's not correct and correct them.\n\n"
                    "Reply EXACTLY in this format:\n"
                    "CRITIQUE: [one sentence evaluation]\n"
                    f"ACTION: [7 integers 0-255]\n\n"
                    f"Example reply:\n"
                    f"CRITIQUE: The z_move is too large for the current distance.\n"
                    f"ACTION: {planner_str}"
                )}
            ]
        }]
        return messages

    def _apply_chat_template(
        self,
        image: Image.Image,
        instruction: str,
        planner_action_token_ids: np.ndarray
    ) -> dict:
        messages = self._make_messages(image, instruction, planner_action_token_ids)
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )


    # ─────────────────────────────────────────
    # input 준비 (action token 붙이기 제거)
    # ─────────────────────────────────────────

    def _prepare_inputs(self, inputs: dict) -> tuple:
        """
        [수정] action token ID를 input_ids에 붙이는 로직 제거.

        제거 이유:
            출력 형식이 <action_N> 토큰 → 정수로 바뀌었으므로
            input_ids에 <action_N>을 붙일 필요 없음.
            플래너 액션은 텍스트(연속값)로 이미 프롬프트에 포함됨.

        반환: (input_ids, attention_mask, pixel_values, prompt_length)
        """
        input_ids      = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        pixel_values   = inputs["pixel_values"].to("cuda")
        prompt_length  = input_ids.shape[1]

        return input_ids, attention_mask, pixel_values, prompt_length


    # ─────────────────────────────────────────
    # Forward (GRPO 학습)
    # ─────────────────────────────────────────

    def forward(
        self,
        cached_inputs: dict,
        new_tokens: torch.Tensor
    ) -> tuple:
        """
        [수정] planner_action_tokens 파라미터 제거
              (프롬프트에 이미 연속값으로 포함되어 있음)

        반환: (logits, prompt_length)
        """
        input_ids, attention_mask, pixel_values, prompt_length = \
            self._prepare_inputs(cached_inputs)

        new_tokens_gpu = new_tokens.unsqueeze(0).to("cuda")
        full_input_ids = torch.cat([input_ids, new_tokens_gpu], dim=1)
        full_mask      = torch.cat([attention_mask, torch.ones_like(new_tokens_gpu)], dim=1)

        outputs = self.smol(
            input_ids=full_input_ids,
            attention_mask=full_mask,
            pixel_values=pixel_values,
        )

        del input_ids, attention_mask, pixel_values, new_tokens_gpu, full_input_ids
        torch.cuda.empty_cache()

        return outputs.logits, prompt_length


    # ─────────────────────────────────────────
    # Generate (LIBERO rollout)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        instruction: str,
        max_new_tokens: int = None
    ) -> tuple:
        if max_new_tokens is None:
            max_new_tokens = self.vram_cfg["max_new_tokens"]

        planner_action_tokens = self.get_planner_action_tokens(image, instruction)
        planner_bin_indices   = self.action_tokenizer.openvla_ids_to_bin_indices(planner_action_tokens)

        cached_inputs = self._apply_chat_template(image, instruction, planner_action_tokens)
        input_ids, attention_mask, pixel_values, input_length = \
            self._prepare_inputs(cached_inputs)

        generated_ids = self.smol.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )
        del input_ids, attention_mask, pixel_values
        torch.cuda.empty_cache()

        new_tokens  = generated_ids[0][input_length:]
        output_text = self.processor.decode(new_tokens, skip_special_tokens=True)
        print("=" * 50)
        print(f"[generate] {output_text[:50]}")

        critique = self._parse_critique(output_text)

        # [수정] planner_bin_indices를 fallback으로 전달
        action_vector, action_token_ids = self._parse_and_decode_action(
            output_text, planner_bin_indices
        )

        self._log_modification(planner_bin_indices, action_token_ids)

        return critique, action_vector, action_token_ids, planner_action_tokens, cached_inputs, new_tokens.cpu()


    # ─────────────────────────────────────────
    # 출력 파싱
    # ─────────────────────────────────────────

    def _parse_critique(self, text: str) -> str:
        """CRITIQUE: ... 또는 [CRITIQUE]...[/CRITIQUE] 모두 허용"""
        try:
            # [CRITIQUE]...[/CRITIQUE] 형식
            m = re.search(r"\[CRITIQUE\](.*?)(\[\/CRITIQUE\]|$)", text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
            # CRITIQUE: ... 형식
            m = re.search(r"CRITIQUE:\s*(.+?)(?=\nACTION:|$)", text, re.DOTALL | re.IGNORECASE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        return ""

    def _parse_and_decode_action(
        self,
        output_text: str,
        planner_bin_indices: np.ndarray
    ) -> tuple:
        """
        [수정] 정수 출력 파서 + lenient fallback.

        파싱 우선순위:
            1. "ACTION: N1 N2 N3 N4 N5 N6 N7" 형식 파싱
            2. 출력 전체에서 유효 정수(0~255) 7개 추출
            3. 부족하면 플래너 값으로 채움 (zeros 대신)

        fallback이 zeros가 아닌 플래너 값인 이유:
            zeros → 로봇이 아무것도 안 함 → 항상 실패 → 보상 없음 → 학습 불가
            플래너 값 → 최소한 OpenVLA 수준 행동 → 일부 성공 가능 → 보상 신호 유지
        """
        smol_action_start = len(self.processor.tokenizer) - 256

        def bins_to_result(bin_list):
            arr = np.clip(np.array(bin_list[:7]), 0, 255)
            token_ids = arr + smol_action_start
            return self.action_tokenizer.decode_token_ids_to_actions(token_ids), token_ids

        try:
            # ① ACTION: 뒤 정수 파싱 (가장 엄격한 형식)
            action_match = re.search(
                r"ACTION:\s*([\d\s]+)",
                output_text, re.IGNORECASE
            )
            if action_match:
                nums = [
                    int(n) for n in re.findall(r"\b(\d{1,3})\b", action_match.group(1))
                    if 0 <= int(n) <= 255
                ]
                if len(nums) >= 7:
                    return bins_to_result(nums)

            # ② [ACTION]...[/ACTION] 사이 정수 파싱
            action_tag = re.search(
                r"\[ACTION\](.*?)(\[\/ACTION\]|$)",
                output_text, re.DOTALL | re.IGNORECASE
            )
            if action_tag:
                nums = [
                    int(n) for n in re.findall(r"\b(\d{1,3})\b", action_tag.group(1))
                    if 0 <= int(n) <= 255
                ]
                if len(nums) >= 7:
                    return bins_to_result(nums)

            # ③ 전체 출력에서 유효 정수 추출 (가장 lenient)
            all_nums = [
                int(n) for n in re.findall(r"\b(\d{1,3})\b", output_text)
                if 0 <= int(n) <= 255
            ]

            if len(all_nums) >= 7:
                return bins_to_result(all_nums)

            # ④ 일부만 있으면 나머지를 플래너 값으로 채움
            if len(all_nums) > 0:
                filled = list(all_nums[:7])
                while len(filled) < 7:
                    filled.append(int(planner_bin_indices[len(filled)]))
                print(f"[파싱 부분성공] {len(all_nums)}개 파싱 → 나머지 플래너 값으로 채움")
                return bins_to_result(filled)

        except Exception as e:
            print(f"[파싱 오류] {e}")

        # ⑤ 완전 실패 → 플래너 값 사용 (zeros 아님)
        print("[파싱 실패] 플래너 값 사용 (zeros 방지)")
        return bins_to_result(list(planner_bin_indices))

    def _log_modification(
        self,
        planner_bin_indices: np.ndarray,
        actor_token_ids: np.ndarray
    ) -> None:
        smol_action_start = len(self.processor.tokenizer) - 256
        actor_bins = actor_token_ids - smol_action_start

        if np.array_equal(planner_bin_indices, actor_bins):
            print("[수정 여부] 유지됨")
        else:
            diffs = [
                f"{ACTION_DIM_NAMES[i]}({planner_bin_indices[i]}→{actor_bins[i]})"
                for i in range(7) if planner_bin_indices[i] != actor_bins[i]
            ]
            print(f"[수정 여부] 수정됨: {', '.join(diffs)}")

    def close(self):
        self.planner_socket.close()
        print("Planner 서버 연결 종료")