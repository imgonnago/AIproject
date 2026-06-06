"""
Actor Model (SmolVLM2-500M, Paradigm A - Forward Hook 방식)

[설계 구조]
    image + text → SmolVLM2 정상 처리 (inputs_merger 포함)
                                    ↓
                              text_model 진입 직전
                              ← pre-hook으로 action_embeds 삽입
                                    ↓
    [image_embeds + text_embeds + action_embeds] → LLM

[핵심 원리]
    SmolVLM2가 image+text를 inputs_merger로 합치는 과정을 그대로 유지.
    text_model(SmolLM2) 진입 직전에만 Projection 출력 action_embeds를 concat.
    → inputs_merger 충돌 없음
    → Projection gradient 흐름 유지 (forward()에서 gradient 계산)
    → pixel_values를 직접 사용하므로 image_hidden_states 형식 문제 없음

[학습 컴포넌트]
    LoRA (q/k/v)             : SmolVLM2 LLM attention 학습
    Projection               : GRPO loss → logits → LLM → action_embeds → Projection
    action token embedding   : sparse hook으로 <action_N> 256행 학습

[OOM 방지]
    vision encoder frozen + pixel_values.detach()
    → vision encoder 중간 활성화 backward 전 즉시 해제
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
        cfg = dict(group_size=2, max_new_tokens=40, lora_r=4, lora_alpha=8, use_8bit_adam=True, label="12GB")
    elif total_gb >= 7:
        cfg = dict(group_size=2, max_new_tokens=30, lora_r=4, lora_alpha=8, use_8bit_adam=True, label="8GB")
    else:
        cfg = dict(group_size=2, max_new_tokens=20, lora_r=4, lora_alpha=8, use_8bit_adam=True, label="6GB")

    print(f"[VRAM 설정] {cfg['label']}: group={cfg['group_size']}, "
          f"max_new={cfg['max_new_tokens']}, lora_r={cfg['lora_r']}")
    return cfg

SFT_MODEL_PATH = "checkpoints/sft_stage2_5" # SmolVLM2-500M 4bit 사전학습 모델 경로
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
        self.processor = AutoProcessor.from_pretrained(SFT_MODEL_PATH)

        # ── 2. Projection ─────────────────────────────
        # Paradigm A의 핵심 학습 컴포넌트
        # OpenVLA embedding space(4096) → SmolVLM2 LLM space(960) 번역
        # GRPO 학습을 통해 이 번역 능력이 개선됨
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
            SFT_MODEL_PATH,
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

        # Vision encoder freeze: OOM 방지 (detach + frozen → backward 없음)
        for name, param in self.smol.named_parameters():
            if "vision" in name.lower() or "visual" in name.lower():
                param.requires_grad_(False)

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

        # ── 7. Action embedding 삽입용 pre-hook ──────
        # 핵심: inputs_merger(image+text 합침) 이후,
        #        text_model(LLM) 진입 직전 시점에 action_embeds 삽입
        # → image/text/action 세 가지가 모두 LLM에 전달됨
        self._action_embeds_buffer = [None]  # hook closure용 mutable container
        self._action_insert_pos   = [None]  # None=끝에 append, int=해당 위치에 insert
        text_model = self.smol.base_model.model.model.text_model  # SmolLM2 backbone
        text_model.register_forward_pre_hook(
            self._action_injection_hook,
            with_kwargs=True  # PyTorch 2.0+ 필요
        )
        print("[action hook] text_model pre-hook 등록 완료")

        # ── connector hook: generate() 중 image features 캡처 ──
        # generate()에서 SmolVLM2가 내부적으로 pixel_values → connector 처리한 결과를 캡처
        # → forward()(학습)에서 이 features를 image_hidden_states로 재사용
        # → 학습 시 vision encoder 재실행 없음 → OOM 방지
        # → inputs_merger가 기대하는 정확한 형식 보장
        self._ihs_cache    = [None]   # 캡처된 image features
        self._capture_ihs  = False    # generate() 중에만 캡처 활성화
        connector = self.smol.base_model.model.model.connector
        connector.register_forward_hook(self._connector_output_hook)
        print("[connector hook] image features 캡처 hook 등록 완료")

        # ── 8. ZeroMQ ────────────────────────────────
        zmq_context = zmq.Context()
        self.planner_socket = zmq_context.socket(zmq.REQ)
        self.planner_socket.connect(f"tcp://{PLANNER_HOST}:{PLANNER_PORT}")
        print("ActorModel initialization complete!")

    def _register_action_only_embedding_hook(self):
        smol_action_start = len(self.processor.tokenizer) - 256
        
        def _sparse_hook(grad):
            sparse = torch.zeros_like(grad)
            sparse[smol_action_start:] = grad[smol_action_start:]
            return sparse

        # Input Embeddings 학습 허용
        in_emb = self.smol.get_input_embeddings()
        in_emb.weight.requires_grad_(True)
        in_emb.weight.register_hook(_sparse_hook)

        # Output Embeddings(lm_head) 학습 허용 (필수!)
        out_emb = self.smol.get_output_embeddings()
        if out_emb is not None:
            out_emb.weight.requires_grad_(True)
            out_emb.weight.register_hook(_sparse_hook)

        print(f"[sparse hook] 입력/출력 action embedding 행({smol_action_start}~)만 학습")

    def _connector_output_hook(self, module, input, output):
        """
        SmolVLM2 connector(pixel shuffle) 출력을 캡처.

        캡처 시점: generate() 내부에서 pixel_values → vision_model → connector 처리 후
        캡처 결과: inputs_merger가 기대하는 정확한 형식의 image_hidden_states
            shape: (1, n_compressed_patches, hidden_size)
        
        이 features를 training forward()에서 image_hidden_states로 넘기면
        inputs_merger가 동일한 포맷으로 인식 → 검증 통과
        """
        if self._capture_ihs:
            self._ihs_cache[0] = output.detach().cpu()

    def _action_injection_hook(self, module, args, kwargs):
        """
        [Paradigm A 핵심] SmolLM2 transformer 진입 직전 호출.

        호출 시점:
            SmolVLMModel.forward():
                image_features = get_image_features(pixel_values)
                merged = inputs_merger(input_ids, embeds, image_features)  ← image+text 합침
                text_model(inputs_embeds=merged, ...)  ← 여기 진입 직전에 hook 실행

        삽입 조건: prefill 스텝에서만 삽입 (past_key_values=None)
            generate() prefill:  action_embeds 삽입 → KV cache에 저장 → decode는 캐시 사용
            forward() training:  단일 pass → action_embeds 삽입
            generate() decode:   past_key_values 있음 → 삽입 안 함 (이미 캐시에 있음)

        결과:
            inputs_embeds: (1, seq, 960) → (1, seq+7, 960)
            attention_mask: (1, seq) → (1, seq+7)
        """
        action_embeds = self._action_embeds_buffer[0]
        if action_embeds is None:
            return args, kwargs

        # decode 스텝이면 이미 KV cache에 action이 있으므로 스킵
        if kwargs.get("past_key_values") is not None:
            return args, kwargs

        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            return args, kwargs

        # action_embeds 삽입: insert_pos 위치에 따라 append 또는 positional insert
        # None  → generate() 레거시: 프롬프트 끝에 append
        # int   → forward()/sft/generate() 모드: 지정 위치에 insert
        #         ex) insert_pos = seq - critique_token_len
        #             → [text | action_7 | CRITIQUE: ] 순서 보장
        insert_pos = self._action_insert_pos[0]
        mask = kwargs.get("attention_mask")
        extra = torch.ones(1, 7, dtype=embeds.dtype if mask is None else mask.dtype,
                           device=embeds.device)

        if insert_pos is None:
            # 끝에 append: (1, seq, 960) + (1, 7, 960) → (1, seq+7, 960)
            kwargs["inputs_embeds"] = torch.cat(
                [embeds, action_embeds.to(embeds.dtype)], dim=1
            )
            if mask is not None:
                kwargs["attention_mask"] = torch.cat([mask, extra], dim=1)
        else:
            # 지정 위치에 insert: [embeds[:pos] | action_7 | embeds[pos:]]
            kwargs["inputs_embeds"] = torch.cat([
                embeds[:, :insert_pos, :],
                action_embeds.to(embeds.dtype),
                embeds[:, insert_pos:, :]
            ], dim=1)
            if mask is not None:
                kwargs["attention_mask"] = torch.cat([
                    mask[:, :insert_pos],
                    extra,
                    mask[:, insert_pos:]
                ], dim=1)

        # position_ids 확장 (제공된 경우): 연속적인 위치 추가
        pos_ids = kwargs.get("position_ids")
        if pos_ids is not None:
            last_pos = pos_ids[0, -1].item()
            extra_pos = torch.arange(
                last_pos + 1, last_pos + 8,
                dtype=pos_ids.dtype, device=pos_ids.device
            ).unsqueeze(0)
            kwargs["position_ids"] = torch.cat([pos_ids, extra_pos], dim=1)

        return args, kwargs


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
    # 프롬프트 헬퍼
    # ─────────────────────────────────────────

    def _make_action_text(self, planner_bin_indices: np.ndarray) -> str:
        action_vec = self.action_tokenizer.bin_indices_to_continuous(planner_bin_indices)
        x, y, z, roll, pitch, yaw, gripper = action_vec
        return (
            f"x={x:+.2f}, y={y:+.2f}, z={z:+.2f}, "
            f"roll={roll:+.2f}, pitch={pitch:+.2f}, yaw={yaw:+.2f}, "
            f"gripper={'open' if gripper > 0 else 'close'}"
        )

    def _make_messages_pass1(
        self,
        image: Image.Image,
        instruction: str,
        planner_bin_indices: np.ndarray
    ) -> list:
        """
        Pass 1: 이미지+플래너 액션을 보고 한 문장 비판 생성.
        출력: 순수 텍스트 (액션 토큰 없음)
        """
        action_text = self._make_action_text(planner_bin_indices)
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Task: {instruction}\n"
                    f"Proposed: {action_text}\n"
                    f"Briefly critique the proposed action in one sentence."
                )}
            ]
        }]

    def _make_messages_pass2(
        self,
        image: Image.Image,
        instruction: str,
        planner_bin_indices: np.ndarray,
        critique_text: str
    ) -> list:
        """
        Pass 2: 비판 텍스트를 컨텍스트로 받아 수정된 액션 토큰 생성.
        출력: <action_N>×7 + EOS
        """
        action_text  = self._make_action_text(planner_bin_indices)
        planner_str  = " ".join([f"<action_{b}>" for b in planner_bin_indices])
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Task: {instruction}\n"
                    f"Critique: {critique_text}\n"
                    f"Output exactly 7 corrected action tokens:\n"
                    f"{planner_str}"
                )}
            ]
        }]

    # 레거시 호환용 (stage_3 capture_ihs 등에서 사용)
    def _make_messages(self, image, instruction, planner_bin_indices):
        return self._make_messages_pass2(image, instruction, planner_bin_indices,
                                         "I observe the scene and evaluate the proposed action.")

    def _apply_chat_template(
        self,
        image: Image.Image,
        instruction: str,
        planner_bin_indices: np.ndarray,
        critique_text: str = "I observe the scene and evaluate the proposed action."
    ) -> dict:
        """Pass 2 캐시 입력 생성 (critique_token_len=0, action embeds는 끝에 삽입)."""
        return self._apply_chat_template_pass2(image, instruction, planner_bin_indices, critique_text)

    def _apply_chat_template_pass2(
        self,
        image: Image.Image,
        instruction: str,
        planner_bin_indices: np.ndarray,
        critique_text: str
    ) -> dict:
        """
        Pass 2 프롬프트 토크나이즈.
        critique_token_len=0 → action embeds가 프롬프트 끝에 삽입됨.
        """
        messages = self._make_messages_pass2(image, instruction, planner_bin_indices, critique_text)
        cached = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        cached["critique_token_len"] = 0  # action embeds → 프롬프트 끝에 삽입
        return cached

    # ─────────────────────────────────────────
    # Pass 1: 비판 텍스트 생성
    # ─────────────────────────────────────────

    @torch.no_grad()
    def generate_critique(
        self,
        image: Image.Image,
        instruction: str,
        planner_bin_indices: np.ndarray
    ) -> str:
        """
        Pass 1: 이미지+명령어+플래너 액션 → 한 문장 비판 텍스트.
        생성 후 VRAM 해제 → Pass 2에서 재사용 없음.
        """
        messages = self._make_messages_pass1(image, instruction, planner_bin_indices)
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        try:
            out = self.smol.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            prompt_len = inputs["input_ids"].shape[1]
            critique = self.processor.decode(
                out[0][prompt_len:], skip_special_tokens=True
            ).strip()
            del out
        except Exception:
            critique = ""

        del inputs
        torch.cuda.empty_cache()
        return critique if critique else "The proposed action seems reasonable."

    # ─────────────────────────────────────────
    # SFT용 IHS 캡처 (Pass 2 형식, Pass 1 없음)
    # ─────────────────────────────────────────

    @torch.no_grad()
    def capture_for_sft(
        self,
        image: Image.Image,
        instruction: str,
        critique_text: str
    ) -> tuple:
        """
        Stage 2/2.5 SFT 전용: precomputed critique로 Pass 2 IHS 캡처.
        Pass 1 실행 없이 바로 Pass 2 형식으로 IHS + cached_inputs 반환.
        ZeroMQ로 planner_tokens도 함께 수집.
        """
        planner_action_tokens = self.get_planner_action_tokens(image, instruction)
        planner_bin_indices   = self.action_tokenizer.openvla_ids_to_bin_indices(
            np.array(planner_action_tokens)
        )
        cached_inputs = self._apply_chat_template_pass2(
            image, instruction, planner_bin_indices, critique_text
        )

        input_ids      = cached_inputs["input_ids"].to("cuda")
        attention_mask = cached_inputs["attention_mask"].to("cuda")
        pixel_values   = cached_inputs["pixel_values"].to("cuda", dtype=torch.bfloat16).detach()
        pixel_values.requires_grad_(False)

        action_embeds = self.action_tokenizer.embed_action_tokens(
            torch.tensor(planner_action_tokens, dtype=torch.long)
        )

        try:
            self._action_embeds_buffer[0] = action_embeds
            self._action_insert_pos[0]    = input_ids.shape[1]  # 끝에 삽입
            self._capture_ihs  = True
            self._ihs_cache[0] = None
            self.smol.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        finally:
            self._action_embeds_buffer[0] = None
            self._action_insert_pos[0]    = None
            self._capture_ihs = False

        ihs = self._ihs_cache[0]
        del input_ids, attention_mask, pixel_values
        torch.cuda.empty_cache()

        return planner_action_tokens, cached_inputs, ihs


    # ─────────────────────────────────────────
    # Forward (GRPO 학습)
    # ─────────────────────────────────────────

    def forward(
        self,
        cached_inputs: dict,
        new_tokens: torch.Tensor,
        planner_action_tokens: np.ndarray,
        image_hidden_states: torch.Tensor   # generate()에서 캡처된 connector 출력 (CPU)
    ) -> tuple:
        """
        Projection gradient 흐름:
            embed_action_tokens() → action_embeds (Projection 통과)
            → _action_embeds_buffer에 저장
            → hook이 text_model 직전에 inputs_embeds에 concat
            → loss → logits → LLM → action_embeds → Projection ← gradient

        prompt_length = input_ids.shape[1] + 7
            (hook이 action embeddings 7개를 삽입하므로 logit 슬라이싱 위치 보정)

        :return: (logits, prompt_length)
        """
        input_ids      = cached_inputs["input_ids"].to("cuda")
        attention_mask = cached_inputs["attention_mask"].to("cuda")

        # Projection으로 action_embeds 계산 (gradient 흐름 유지)
        # → _action_injection_hook이 insert_pos 위치(=prompt 끝)에 삽입
        action_embeds = self.action_tokenizer.embed_action_tokens(
            torch.tensor(planner_action_tokens, dtype=torch.long)
        )  # (1, 7, 960)

        # prompt_length: original prompt + 7 (hook이 prompt 끝에 삽입)
        # LLM이 보는 sequence: [prompt | action_7 | new_tokens]
        # gen_logits = logits[0, prompt_length-1:-1] → (N, vocab) ✓
        prompt_length = input_ids.shape[1] + 7

        # image_hidden_states: generate()의 connector hook이 캡처한 features
        # SmolVLM2 내부 처리와 동일한 형식 → inputs_merger 검증 통과
        # vision encoder 재실행 없음 → OOM 방지
        ihs_gpu = image_hidden_states.to("cuda")

        new_tokens_gpu = new_tokens.unsqueeze(0).to("cuda")
        full_ids  = torch.cat([input_ids, new_tokens_gpu], dim=1)
        full_mask = torch.cat([
            attention_mask,
            torch.ones(1, new_tokens.shape[0], dtype=torch.long, device="cuda")
        ], dim=1)

        try:
            self._action_embeds_buffer[0] = action_embeds
            # "CRITIQUE: " 앞에 action_7 삽입 (generate()와 동일한 구조)
            # [image | text | action_7 | CRITIQUE: | new_tokens] 순서로 학습
            critique_token_len            = cached_inputs.get("critique_token_len", 0)
            self._action_insert_pos[0]    = input_ids.shape[1] - critique_token_len
            outputs = self.smol(
                input_ids=full_ids,
                attention_mask=full_mask,
                pixel_values=None,
                image_hidden_states=ihs_gpu,
            )
        finally:
            self._action_embeds_buffer[0] = None
            self._action_insert_pos[0]    = None

        del input_ids, attention_mask, ihs_gpu, new_tokens_gpu, full_ids, full_mask
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
        """
        두 패스 추론:
          Pass 1: 이미지+명령어+플래너 액션 → 비판 텍스트 (VRAM 해제)
          Pass 2: 비판 텍스트 컨텍스트 + action embeds → 수정된 액션 토큰 7개

        반환: 8-tuple (critique, action_vector, action_token_ids,
                        planner_action_tokens, cached_inputs,
                        new_tokens(CPU), parsing_failed, image_hidden_states)
        """
        if max_new_tokens is None:
            max_new_tokens = self.vram_cfg["max_new_tokens"]

        planner_action_tokens = self.get_planner_action_tokens(image, instruction)
        planner_bin_indices   = self.action_tokenizer.openvla_ids_to_bin_indices(
            planner_action_tokens
        )

        # ── Pass 1: 비판 텍스트 생성 ─────────────────────
        critique = self.generate_critique(image, instruction, planner_bin_indices)
        # generate_critique() 내부에서 VRAM 해제 완료

        # ── Pass 2: 수정된 액션 토큰 생성 ────────────────
        cached_inputs = self._apply_chat_template_pass2(
            image, instruction, planner_bin_indices, critique
        )

        input_ids      = cached_inputs["input_ids"].to("cuda")
        attention_mask = cached_inputs["attention_mask"].to("cuda")
        pixel_values   = cached_inputs["pixel_values"].to("cuda", dtype=torch.bfloat16).detach()
        pixel_values.requires_grad_(False)

        action_embeds = self.action_tokenizer.embed_action_tokens(
            torch.tensor(planner_action_tokens, dtype=torch.long)
        )

        # critique_token_len=0 → action embeds는 프롬프트 끝에 삽입
        insert_pos    = input_ids.shape[1]
        prompt_length = input_ids.shape[1]

        try:
            self._action_embeds_buffer[0] = action_embeds
            self._action_insert_pos[0]    = insert_pos
            self._capture_ihs  = True
            self._ihs_cache[0] = None
            generated_ids = self.smol.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        finally:
            self._action_embeds_buffer[0] = None
            self._action_insert_pos[0]    = None
            self._capture_ihs = False

        image_hidden_states = self._ihs_cache[0]

        del input_ids, attention_mask, pixel_values
        torch.cuda.empty_cache()

        new_tokens = generated_ids[0][prompt_length:]
        del generated_ids
        torch.cuda.empty_cache()

        output_text = self.processor.decode(new_tokens, skip_special_tokens=False)
        print("=" * 50)
        print(f"[Pass 1] CRITIQUE: {critique[:80]}")
        print(f"[Pass 2] ACTION:   {output_text}")

        action_vector, action_token_ids, parsing_failed = self._parse_and_decode_action(new_tokens)
        self._log_modification(planner_bin_indices, action_token_ids)

        return (critique, action_vector, action_token_ids,
                planner_action_tokens, cached_inputs,
                new_tokens.cpu(), parsing_failed, image_hidden_states)


    # ─────────────────────────────────────────
    # 출력 파싱
    # ─────────────────────────────────────────

    def _parse_critique(self, text: str) -> str:
        # Pass 1 출력은 순수 텍스트이므로 그대로 반환
        return text.strip() if text else ""

    def _parse_and_decode_action(self, new_tokens: torch.Tensor) -> tuple:
        """token ID 범위로 <action_N> 탐지."""
        smol_action_start = len(self.processor.tokenizer) - 256
        fallback_bins = np.full(7, 128)

        def bins_to_result(bin_list, failed=False):
            arr = np.clip(np.array(bin_list[:7]), 0, 255)
            token_ids = arr + smol_action_start
            return self.action_tokenizer.decode_token_ids_to_actions(token_ids), token_ids, failed

        new_tokens_np = new_tokens.cpu().numpy()

        # ① token ID 범위로 직접 탐지
        action_mask = (
            (new_tokens_np >= smol_action_start) &
            (new_tokens_np < smol_action_start + 256)
        )
        action_ids_found = new_tokens_np[action_mask]
        if len(action_ids_found) >= 7:
            return bins_to_result(action_ids_found[:7] - smol_action_start, failed=False)

        # ② 텍스트 fallback
        output_text = self.processor.decode(new_tokens, skip_special_tokens=False)
        m = re.search(r"\[ACTION\](.*?)(\[\/ACTION\]|$)", output_text, re.DOTALL | re.IGNORECASE)
        if m:
            indices = re.findall(r"<action_(\d+)>", m.group(1))
            if len(indices) == 7:
                return bins_to_result([int(i) for i in indices], failed=False)

        print("[파싱 실패] <action_N> 미생성 → cold start 페널티 대상")
        return bins_to_result(fallback_bins, failed=True)

    def _log_modification(self, planner_bin_indices, actor_token_ids):
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
        

    def load_and_scale_checkpoint(self, path: str, scale: float = 0.5):
        """
        SFT 체크포인트 로드 후 action lm_head 스케일 재적용.

        Stage 2.5 과훈련으로 action token logit이 높아진 상태에서
        GRPO 시작 시 텍스트 생성 능력 회복을 위해 사용.

        사용법 (smol_train.py):
            actor = ActorModel()
            actor.load_and_scale_checkpoint("checkpoints/sft_stage2_5")
        """
        from safetensors.torch import load_file

        if not os.path.exists(path):
            print(f"[체크포인트] {path} 없음 → 베이스 모델 사용")
            return

        sf_files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if not sf_files:
            print(f"[체크포인트] safetensors 없음 → 베이스 모델 사용")
            return

        state_dict = {}
        for f in sf_files:
            state_dict.update(load_file(os.path.join(path, f)))

        missing, unexpected = self.smol.load_state_dict(state_dict, strict=False)
        print(f"[체크포인트 로드] {path}")
        print(f"  완료 | missing={len(missing)} unexpected={len(unexpected)}")

        # 추가
        action_emb_path = os.path.join(path, "action_embeddings.pt")
        if os.path.exists(action_emb_path):
            action_emb = torch.load(action_emb_path, map_location="cpu")
            dtype = self.smol.get_input_embeddings().weight.dtype
            with torch.no_grad():
                self.smol.get_input_embeddings().weight.data[-256:] = \
                    action_emb['input_embeddings'].to("cuda").to(dtype)
                self.smol.get_output_embeddings().weight.data[-256:] = \
                    action_emb['output_embeddings'].to("cuda").to(dtype)
            print("[action_embeddings] 로드 완료")

        # 체크포인트가 덮어쓴 lm_head에 스케일 재적용
        if scale < 1.0:
            out_emb = self.smol.get_output_embeddings()
            if out_emb is not None:
                with torch.no_grad():
                    out_emb.weight.data[-256:] *= scale
                after   = out_emb.weight.data[-256:].norm(dim=1).mean().item()
                regular = out_emb.weight.data[:-256].norm(dim=1).mean().item()
                print(f"[scale ×{scale}] text norm: {regular:.3f} | action norm: {after:.3f}")


    def close(self):
        self.planner_socket.close()
        print("Planner 서버 연결 종료")