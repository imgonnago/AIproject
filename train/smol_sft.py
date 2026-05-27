"""
smol_sft.py — 형식 학습용 SFT (Supervised Fine-Tuning)

[목적]
    cold start 문제 해결: GRPO 시작 전에 올바른 출력 형식 학습
    CRITIQUE 텍스트 + [ACTION] <action_N>×7 [/ACTION] 형식 안정화

[데이터 전략]
    실제 환경/데이터 불필요. 합성 데이터 사용:
        - 이미지:         224×224 단색 blank (GRPO와 동일한 이미지 컨텍스트 구조)
        - 플래너 action:  랜덤 bin indices (0~255)
        - 타겟 출력:      올바른 형식으로 플래너 action 그대로 출력
                          → 모델이 먼저 형식 학습 → GRPO에서 수정 능력 학습

[핵심 변경]
    이미지 포함 SFT: blank 이미지 IHS를 1회 캡처 → 모든 스텝 재사용
    GRPO와 동일한 입력 분포 [image_tokens | text | action_7 | target] 학습
    OOM 없음: vision encoder는 캡처 시 1회만 실행, 이후 image_hidden_states 재사용

[학습 후 기대 효과]
    파싱 성공률 ~0% → ~90%+
    GRPO cold start 시 reward 신호 즉시 활성화

실행:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/smol_sft.py
"""

import os
import sys
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256"
)
sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from SmolVLM_actor.smol_actor_model import ActorModel

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

NUM_STEPS    = 700         # SFT 스텝 수 (수 분 소요)
LEARNING_RATE = 5e-5       # GRPO보다 작게: 형식만 학습, 언어 능력 보존
SAVE_PATH    = "checkpoints/sft"

# 형식 학습용 CRITIQUE 템플릿 (다양성 확보)
CRITIQUE_TEMPLATES = [
    "The proposed action looks reasonable.",
    "The action appears correct for this task.",
    "The proposed action seems appropriate.",
    "The action is suitable for the current state.",
    "The proposed movement looks correct.",
]

# SFT 학습용 태스크 설명 (실제 LIBERO 태스크와 유사하게)
TASK_INSTRUCTIONS = [
    "pick up the object and place it in the basket",
    "put the soup in the basket",
    "move the object to the target location",
    "grasp the item and place it correctly",
    "pick up and place the object",
]



# ─────────────────────────────────────────
# image features 사전 캡처 (1회)
# ─────────────────────────────────────────

def capture_shared_ihs(actor: ActorModel) -> torch.Tensor:
    """
    SFT 시작 전 blank 이미지로 image features 1회 캡처.
    SFT 전 스텝에서 image_hidden_states로 재사용.

    vision encoder는 이 1회만 실행 → OOM 없음.
    IHS를 공유하므로 모든 스텝이 동일한 이미지 컨텍스트를 가짐.
    이미지 내용(색상)이 아니라 이미지 컨텍스트 구조가 목적이므로 OK.
    """
    print("[SFT] blank 이미지로 image features 사전 캡처 중...")
    blank       = Image.new("RGB", (224, 224), color=(128, 128, 128))
    instruction = "pick up the object"

    with torch.no_grad():
        # actor.generate()는 8-tuple 반환, 마지막이 image_hidden_states (CPU tensor)
        *_, shared_ihs = actor.generate(image=blank, instruction=instruction)

    torch.cuda.empty_cache()
    print(f"[SFT] image features 캡처 완료: shape={shared_ihs.shape}")
    return shared_ihs  # CPU tensor, 모든 SFT 스텝에서 재사용


# ─────────────────────────────────────────
# 합성 데이터 생성
# ─────────────────────────────────────────

def make_sft_example(actor: ActorModel, blank_image: Image.Image):
    """
    합성 SFT 데이터 1개 생성.

    입력:
        blank 이미지(고정) + 랜덤 태스크 + 랜덤 플래너 action
        이미지를 포함하여 GRPO와 동일한 입력 분포 구성:
            [image_tokens | text_tokens | action_7 | target]
        vision encoder는 capture_shared_ihs()에서 1회만 실행 → OOM 없음

    타겟:
        CRITIQUE: [템플릿 문장]
        [ACTION] <action_X> × 7 [/ACTION]

    반환:
        (cached_inputs, planner_action_tokens, target_ids)
    """
    instruction           = TASK_INSTRUCTIONS[np.random.randint(len(TASK_INSTRUCTIONS))]
    planner_bin_indices   = np.random.randint(0, 256, 7)
    planner_action_tokens = (32000 - planner_bin_indices).astype(np.int64)

    critique_text = CRITIQUE_TEMPLATES[np.random.randint(len(CRITIQUE_TEMPLATES))]
    action_str    = " ".join([f"<action_{b}>" for b in planner_bin_indices])
    eos_token     = actor.processor.tokenizer.eos_token
    target_text   = f"CRITIQUE: {critique_text}\n\n[ACTION] {action_str} [/ACTION]{eos_token}"

    # GRPO와 동일한 이미지 기반 프롬프트 (_apply_chat_template 재사용)
    # pixel_values는 cached_inputs에 있지만 sft_step에서 사용 안 함 (shared_ihs 사용)
    cached_inputs = actor._apply_chat_template(blank_image, instruction, planner_bin_indices)

    target_ids = actor.processor.tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"]  # (1, T)

    return cached_inputs, planner_action_tokens, target_ids


# ─────────────────────────────────────────
# SFT 1 스텝
# ─────────────────────────────────────────

def sft_step(actor, optimizer, cached_inputs, planner_action_tokens, target_ids, shared_ihs):
    """
    Cross-entropy loss on target tokens.

    LLM이 보는 sequence (GRPO generate와 동일한 구조):
        [image_tokens | text_tokens | action_7 (hook) | target_T]

    shared_ihs: capture_shared_ihs()에서 캡처된 blank 이미지 image features (CPU)
                inputs_merger가 image placeholder 위치에 삽입 → image context 제공
                vision encoder 재실행 없음 → OOM 없음
    """
    optimizer.zero_grad()

    input_ids      = cached_inputs["input_ids"].to("cuda")
    attention_mask = cached_inputs["attention_mask"].to("cuda")
    target_ids_gpu = target_ids.to("cuda")
    T              = target_ids_gpu.shape[1]

    prompt_seq_len = input_ids.shape[1]
    prompt_length  = prompt_seq_len + 7  # +7: hook이 prompt 끝에 삽입할 action embeds

    # Projection으로 action embeddings 계산 (gradient 흐름)
    action_embeds = actor.action_tokenizer.embed_action_tokens(
        torch.tensor(planner_action_tokens, dtype=torch.long)
    )  # (1, 7, 960)

    full_ids  = torch.cat([input_ids, target_ids_gpu], dim=1)   # (1, seq+T)
    full_mask = torch.cat([
        attention_mask,
        torch.ones(1, T, dtype=torch.long, device="cuda")
    ], dim=1)

    ihs_gpu = shared_ihs.to("cuda")

    try:
        actor._action_embeds_buffer[0] = action_embeds
        actor._action_insert_pos[0]    = prompt_seq_len  # prompt 끝 ~ target 사이에 삽입
        outputs = actor.smol(
            input_ids=full_ids,
            attention_mask=full_mask,
            pixel_values=None,
            image_hidden_states=ihs_gpu,   # connector 캡처 IHS → vision encoder 미실행
        )
    finally:
        actor._action_embeds_buffer[0] = None
        actor._action_insert_pos[0]    = None

    # LLM sequence: [image+text (seq) | action_7 | target_T]
    # logits[0, prompt_length-1:-1] → (T, vocab): 각 타겟 토큰 예측
    gen_logits = outputs.logits[0, prompt_length - 1 : -1, :]
    loss       = F.cross_entropy(gen_logits, target_ids_gpu[0])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer.step()

    loss_val = loss.item()
    del input_ids, attention_mask, target_ids_gpu, full_ids, full_mask, ihs_gpu
    del action_embeds, outputs, gen_logits, loss
    torch.cuda.empty_cache()

    return loss_val


# ─────────────────────────────────────────
# SFT 메인
# ─────────────────────────────────────────

def sft_train(actor: ActorModel):
    """
    형식 SFT 학습 루프.
    매 스텝 새 합성 데이터 생성 → 무한 데이터 효과.
    """
    print(f"\n{'='*60}")
    print(f"형식 SFT 시작: {NUM_STEPS} 스텝")
    print(f"목표: CRITIQUE + [ACTION] <action_N>×7 형식 학습")
    print(f"{'='*60}\n")

    # Optimizer: GRPO용보다 낮은 lr, action 임베딩 행 포함
    trainable = list(filter(lambda p: p.requires_grad, actor.parameters()))
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable, lr=LEARNING_RATE)
        print("[optimizer] AdamW8bit")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
        print("[optimizer] AdamW")

    # blank 이미지 + IHS 사전 준비 (1회)
    blank_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    shared_ihs  = capture_shared_ihs(actor)

    losses = []

    for step in range(1, NUM_STEPS + 1):
        # 매 스텝 새 합성 데이터 생성 (blank_image 고정, 내용 무관)
        cached_inputs, planner_action_tokens, target_ids = make_sft_example(actor, blank_image)

        loss = sft_step(actor, optimizer, cached_inputs, planner_action_tokens, target_ids, shared_ihs)
        losses.append(loss)

        if step % 50 == 0:
            avg = sum(losses[-50:]) / 50
            print(f"  [Step {step:4d}/{NUM_STEPS}] loss={avg:.4f}")

            # 형식 학습 확인: 샘플 출력
            if step % 100 == 0:
                _check_format(actor)

    # 저장
    os.makedirs(SAVE_PATH, exist_ok=True)
    actor.smol.save_pretrained(SAVE_PATH)
    actor.processor.save_pretrained(SAVE_PATH)
    print(f"\nSFT 완료! 저장: {SAVE_PATH}")
    print("다음 단계: smol_train.py 실행 (GRPO)")


def _check_format(actor: ActorModel):
    """
    학습 중 형식 출력 확인용.
    actor.generate()로 blank 이미지 → GRPO와 동일한 조건에서 형식 검증.
    에러 나도 무시하고 계속 학습.
    """
    try:
        blank = Image.new("RGB", (224, 224), color=(128, 128, 128))
        (critique, action_vector, action_token_ids, *_) = actor.generate(
            image=blank, instruction="pick up the object"
        )
        smol_start = len(actor.processor.tokenizer) - 256
        format_ok  = any(smol_start <= int(t) < smol_start + 256 for t in action_token_ids)
        print(f"  [형식 확인] CRITIQUE: {critique[:60] if critique else '없음'}")
        print(f"  [형식 확인] action tokens: {'OK' if format_ok else 'FAIL'} | "
              f"action: {np.round(action_vector, 2)}")
    except Exception as e:
        print(f"  [형식 확인 skip] {e}")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main():
    print(f"[CUDA] {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '미설정')}")
    print("Actor 모델 초기화 중... (Planner 서버 불필요)")

    actor = ActorModel()
    actor.smol.gradient_checkpointing_enable()

    alloc = torch.cuda.memory_allocated() / 1024**2
    print(f"[VRAM] 모델 로드 후: {alloc:.0f}MB")

    sft_train(actor)
    actor.close()

if __name__ == "__main__":
    main()