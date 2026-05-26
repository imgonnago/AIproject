"""
smol_sft.py — 형식 학습용 SFT (Supervised Fine-Tuning)

[목적]
    cold start 문제 해결: GRPO 시작 전에 올바른 출력 형식 학습
    CRITIQUE 텍스트 + [ACTION] <action_N>×7 [/ACTION] 형식 안정화

[데이터 전략]
    실제 환경/데이터 불필요. 합성 데이터 사용:
        - 이미지:         224×224 단색 (랜덤 RGB)
        - 플래너 action:  랜덤 bin indices (0~255)
        - 타겟 출력:      올바른 형식으로 플래너 action 그대로 출력
                          → 모델이 먼저 형식 학습 → GRPO에서 수정 능력 학습

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
# 합성 데이터 생성
# ─────────────────────────────────────────

def make_sft_example(actor: ActorModel):
    """
    합성 SFT 데이터 1개 생성 (이미지 없음 → vision encoder 실행 안 함 → OOM 없음).

    입력:
        텍스트 전용 프롬프트 + 랜덤 플래너 action
        이미지 제거 이유: 형식 학습이 목적이므로 시각 정보 불필요
                         vision encoder 실행이 OOM 원인이므로 완전 제거

    타겟:
        CRITIQUE: [템플릿 문장]
        [ACTION] <action_X> × 7 [/ACTION]

        플래너 action을 그대로 복사 → 형식 학습에 집중
        (이후 GRPO에서 언제 수정할지 학습)

    반환:
        (cached_inputs, planner_action_tokens, target_ids)
    """
    instruction = TASK_INSTRUCTIONS[np.random.randint(len(TASK_INSTRUCTIONS))]

    # 랜덤 플래너 bin indices (0~255)
    planner_bin_indices   = np.random.randint(0, 256, 7)
    planner_action_tokens = (32000 - planner_bin_indices).astype(np.int64)

    # 타겟 출력 구성
    critique_text = CRITIQUE_TEMPLATES[np.random.randint(len(CRITIQUE_TEMPLATES))]
    action_str    = " ".join([f"<action_{b}>" for b in planner_bin_indices])
    eos_token = actor.processor.tokenizer.eos_token
    target_text = f"CRITIQUE: {critique_text}\n\n[ACTION] {action_str} [/ACTION]{eos_token}"

    # 텍스트 전용 프롬프트 (image 타입 없음 → pixel_values 생성 안 됨)
    messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"Task: {instruction}\n\n"
                    f"Planner action tokens: {action_str}\n\n"
                    "You are an expert robot action critic. Your job is to critically evaluate the action inferred by the planner.\n"
                    "Judge the situation by looking at the task, the image, and the planner action tokens above.\n"
                    "The action consists of 7 tokens in the format <action_N> (N is 0-255), "
                    "representing: x_move, y_move, z_move, roll, pitch, yaw, and gripper.\n\n"
                    "If the planner's action is correct, output the same tokens. "
                    "If it is wrong, correct the action tokens appropriately.\n\n"
                    "CRITICAL RULE 1: Keep your CRITIQUE extremely short (under 10 words) to save memory.\n"
                    "CRITICAL RULE 2: You MUST output EXACTLY 7 action tokens between [ACTION] and [/ACTION] tags, whether you modified them or not.\n\n"
                    "Reply EXACTLY in this format:\n"
                    "CRITIQUE: [one short sentence evaluation]\n\n"  # 줄바꿈 2번으로 텍스트와 토큰 경계 분리
                    f"[ACTION] {action_str} [/ACTION]\n\n"
                )}
            ]
        }]

    cached_inputs = actor.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    # cached_inputs에 pixel_values 없음 → vision encoder 미실행

    # 타겟 토크나이징
    target_ids = actor.processor.tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"]  # (1, T)

    return cached_inputs, planner_action_tokens, target_ids


# ─────────────────────────────────────────
# SFT 1 스텝
# ─────────────────────────────────────────

def sft_step(actor, optimizer, cached_inputs, planner_action_tokens, target_ids):
    optimizer.zero_grad()

    input_ids      = cached_inputs["input_ids"].to("cuda")
    attention_mask = cached_inputs["attention_mask"].to("cuda")
    target_ids_gpu = target_ids.to("cuda")
    T = target_ids_gpu.shape[1]

    # embedding table에서 직접 조회
    embed_layer   = actor.smol.get_input_embeddings()
    prompt_embeds = embed_layer(input_ids)       # (1, seq, 960)
    target_embeds = embed_layer(target_ids_gpu)  # (1, T, 960)

    # Projection으로 action embeddings
    action_embeds = actor.action_tokenizer.embed_action_tokens(
        torch.tensor(planner_action_tokens, dtype=torch.long)
    )  # (1, 7, 960)

    # 올바른 순서로 직접 조립: [prompt | action_7 | target_T]
    # hook 없이 명시적으로 구성 → 위치 오류 없음
    full_embeds = torch.cat([prompt_embeds, action_embeds, target_embeds], dim=1)
    full_mask   = torch.cat([
        attention_mask,
        torch.ones(1, 7 + T, dtype=torch.long, device="cuda")
    ], dim=1)

    # prompt_length = seq + 7 (action 7개 포함)
    prompt_length = input_ids.shape[1] + 7

    # 이미지 없는 text-only → inputs_merger 검증 통과 (image token 0개)
    outputs = actor.smol(
        inputs_embeds=full_embeds,
        attention_mask=full_mask,
        pixel_values=None,
        image_hidden_states=None,
    )

    gen_logits = outputs.logits[0, prompt_length - 1 : -1, :]  # (T, vocab)
    loss = F.cross_entropy(gen_logits, target_ids_gpu[0])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer.step()

    loss_val = loss.item()
    del input_ids, attention_mask, target_ids_gpu
    del prompt_embeds, action_embeds, target_embeds, full_embeds, full_mask
    del outputs, gen_logits, loss
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

    losses = []

    for step in range(1, NUM_STEPS + 1):
        # 매 스텝 새 합성 데이터 생성
        cached_inputs, planner_action_tokens, target_ids = make_sft_example(actor)

        loss = sft_step(actor, optimizer, cached_inputs, planner_action_tokens, target_ids)
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
    학습 중 형식 출력 확인용 (간단한 generate 테스트).
    에러 나도 무시하고 계속 학습.
    """
    try:
        with torch.no_grad():
            instruction         = "pick up the object"
            planner_bin_indices = np.array([118, 122, 115, 105, 123, 117, 109])
            planner_action_tokens = (32000 - planner_bin_indices).astype(np.int64)
            action_str = " ".join([f"<action_{b}>" for b in planner_bin_indices])

            # 텍스트 전용 프롬프트 (이미지 없음)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Task: {instruction}\n\n"
                        f"Planner action tokens: {action_str}\n\n"
                        "You are an expert robot action critic. Your job is to critically evaluate the action inferred by the planner.\n"
                        "Judge the situation by looking at the task, the image, and the planner action tokens above.\n"
                        "The action consists of 7 tokens in the format <action_N> (N is 0-255), "
                        "representing: x_move, y_move, z_move, roll, pitch, yaw, and gripper.\n\n"
                        "If the planner's action is correct, output the same tokens. "
                        "If it is wrong, correct the action tokens appropriately.\n\n"
                        "CRITICAL RULE 1: Keep your CRITIQUE extremely short (under 10 words) to save memory.\n"
                        "CRITICAL RULE 2: You MUST output EXACTLY 7 action tokens between [ACTION] and [/ACTION] tags, whether you modified them or not.\n\n"
                        "Reply EXACTLY in this format:\n"
                        "CRITIQUE: [one short sentence evaluation]\n\n"  # 줄바꿈 2번으로 텍스트와 토큰 경계 분리
                        f"[ACTION] {action_str} [/ACTION]\n\n"
                    )}
                ]
            }]

            cached = actor.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            )

            input_ids      = cached["input_ids"].to("cuda")
            attention_mask = cached["attention_mask"].to("cuda")
            prompt_length  = input_ids.shape[1]

            action_embeds = actor.action_tokenizer.embed_action_tokens(
                torch.tensor(planner_action_tokens, dtype=torch.long)
            )
            actor._action_embeds_buffer[0] = action_embeds
            actor._action_insert_pos[0]    = None  # generate: append at end

            generated_ids = actor.smol.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=None,           # 이미지 없음
                image_hidden_states=None,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=actor.processor.tokenizer.eos_token_id
            )
            actor._action_embeds_buffer[0] = None
            actor._action_insert_pos[0]    = None

            new_tokens  = generated_ids[0][prompt_length:]
            output_text = actor.processor.decode(new_tokens, skip_special_tokens=False)
            print(f"  [형식 확인] {output_text}")

            del input_ids, attention_mask, generated_ids, new_tokens
            torch.cuda.empty_cache()
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