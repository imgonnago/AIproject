"""
smol_sft.py — 실제 LIBERO 이미지 기반 형식 SFT

[변경사항 - blank 이미지 → 실제 이미지]
    기존: blank 이미지 IHS 1회 캡처 → 전 스텝 재사용
    변경: 매 스텝 실제 LIBERO 이미지 → generate()로 IHS 캡처 후 즉시 재사용

[이미지 다양성 확보]
    TASK_IDS 여러 태스크 로테이션 (기본 5개)
    TASK_SWITCH_EVERY 스텝마다 태스크 전환
    RESET_EVERY 스텝마다 env reset → 동일 태스크 내에서도 다양한 장면

[학습 타겟]
    텍스트: CRITIQUE 템플릿 (다양성)
    액션:   OpenVLA 플래너가 실제 이미지 보고 생성한 실제 액션 토큰
    형식:   CRITIQUE 텍스트\n\n[ACTION] <action_N>×7 [/ACTION]

[OOM 방지]
    generate()로 IHS 캡처 → forward()에서 image_hidden_states로 재사용
    vision encoder 스텝당 1회만 실행

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

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from SmolVLM_actor.smol_actor_model import ActorModel


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

NUM_STEPS         = 800
LEARNING_RATE     = 5e-5
SAVE_PATH         = "checkpoints/sft"

TASK_SUITE        = "libero_10"
TASK_IDS          = [0, 1, 2, 3, 4]   # 다양한 이미지를 위해 여러 태스크 사용
TASK_SWITCH_EVERY = 100                 # N 스텝마다 다음 태스크로 전환
RESET_EVERY       = 15                  # 태스크 내에서 N 스텝마다 env reset
IMG_HEIGHT        = 224
IMG_WIDTH         = 224

# 텍스트 생성 실패 시 fallback (get_scene_text가 빈 문자열 반환할 때만 사용)
FALLBACK_TEXT = "I observe the scene and evaluate the proposed action."


# ─────────────────────────────────────────
# LIBERO 환경
# ─────────────────────────────────────────

def make_env(task_id: int):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_SUITE]()
    task_name      = task_suite.get_task_names()[task_id]
    task_bddl_file = task_suite.get_task_bddl_file_path(task_id)
    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_bddl_file,
        "camera_heights": IMG_HEIGHT,
        "camera_widths":  IMG_WIDTH,
    })
    env.seed(42 + task_id)
    return env, task_name


def get_image(obs: dict) -> Image.Image:
    return Image.fromarray(obs["agentview_image"].astype(np.uint8))


def get_scene_text(actor: ActorModel, image: Image.Image, instruction: str,
                   verbose: bool = False) -> str:
    """
    베이스 SmolVLM2로 장면 설명 텍스트 생성.

    disable_adapter()로 LoRA 가중치 비활성화:
        → 기존 SFT로 액션 토큰 편향된 LoRA 우회
        → 베이스 SmolVLM2 본연의 이미지 이해 능력 사용
        → 추가 VRAM 없음
        → temperature=0.7 → 매 호출마다 다른 자연스러운 텍스트
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Briefly describe the scene for: {instruction}"}
        ]
    }]
    text   = actor.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = actor.processor(
        text=[text], images=[image], return_tensors="pt"
    ).to("cuda")

    try:
        with torch.no_grad():
            # LoRA 비활성화 → 베이스 SmolVLM2 자연스러운 텍스트 생성
            with actor.smol.disable_adapter():
                out = actor.smol.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                )

        desc = actor.processor.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        del out
    except Exception as e:
        print(f"  [scene_text ERROR] {type(e).__name__}: {e}")
        desc = ""

    if verbose or not desc:
        print(f"  [scene_text] '{desc[:80] if desc else '(없음 → fallback 사용)'}'")

    del inputs
    torch.cuda.empty_cache()

    return desc if desc else FALLBACK_TEXT


# ─────────────────────────────────────────
# SFT 데이터 생성 (실제 이미지 기반)
# ─────────────────────────────────────────

def make_sft_example(actor: ActorModel, image: Image.Image, instruction: str):
    """
    실제 LIBERO 이미지로 SFT 데이터 1개 생성.

    generate() 호출로:
        1. 실제 이미지 IHS 캡처
        2. OpenVLA 플래너 실제 액션 토큰 수집
        3. cached_inputs (이미지 포함 프롬프트) 확보

    타겟:
        "CRITIQUE: [템플릿]\n\n[ACTION] <planner_action_N>×7 [/ACTION][EOS]"
        → 실제 이미지에 대한 실제 플래너 액션을 형식에 맞게 출력하도록 학습
    """
    with torch.no_grad():
        (_, _, _, planner_tokens,
         cached_inputs, _, _, ihs) = actor.generate(
            image=image, instruction=instruction
        )

    # 플래너 실제 액션 → bin indices → action 문자열
    planner_bin = actor.action_tokenizer.openvla_ids_to_bin_indices(
        np.array(planner_tokens)
    )  # (7,) 0~255
    action_str = " ".join([f"<action_{b}>" for b in planner_bin])

    # 베이스 SmolVLM2(LoRA 비활성화)로 이미지 보고 자연스러운 텍스트 생성
    # verbose=True: 첫 확인용 (학습 중 매 스텝 출력 원하면 True 유지)
    verbose = not hasattr(make_sft_example, "_logged") or make_sft_example._count < 5
    if not hasattr(make_sft_example, "_count"):
        make_sft_example._count = 0
    make_sft_example._count += 1

    critique_text = get_scene_text(actor, image, instruction, verbose=verbose)
    eos_token     = actor.processor.tokenizer.eos_token
    target_text   = f"{critique_text}\n\n[ACTION] {action_str} [/ACTION]{eos_token}"

    target_ids = actor.processor.tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"]  # (1, T)

    return cached_inputs, planner_tokens, target_ids, ihs


# ─────────────────────────────────────────
# SFT 1 스텝
# ─────────────────────────────────────────

def sft_step(actor, optimizer, cached_inputs, planner_tokens, target_ids, ihs):
    """
    Cross-entropy loss on target tokens.

    LLM 시퀀스 (GRPO와 동일):
        [image_tokens | text_tokens | action_7 (hook) | CRITIQUE: | target_T]

    ihs: 이 스텝의 실제 이미지에서 캡처한 IHS
         vision encoder 재실행 없음 → OOM 없음
    """
    optimizer.zero_grad()

    input_ids      = cached_inputs["input_ids"].to("cuda")
    attention_mask = cached_inputs["attention_mask"].to("cuda")
    target_ids_gpu = target_ids.to("cuda")
    T              = target_ids_gpu.shape[1]

    prompt_seq_len    = input_ids.shape[1]
    critique_tok_len  = cached_inputs.get("critique_token_len", 0)
    prompt_length     = prompt_seq_len + 7  # +7: hook action embeds

    # Projection → action embeddings (gradient 흐름)
    action_embeds = actor.action_tokenizer.embed_action_tokens(
        torch.tensor(planner_tokens, dtype=torch.long)
    )  # (1, 7, 960)

    full_ids  = torch.cat([input_ids, target_ids_gpu], dim=1)
    full_mask = torch.cat([
        attention_mask,
        torch.ones(1, T, dtype=torch.long, device="cuda")
    ], dim=1)

    ihs_gpu = ihs.to("cuda")

    try:
        actor._action_embeds_buffer[0] = action_embeds
        actor._action_insert_pos[0]    = prompt_seq_len - critique_tok_len
        outputs = actor.smol(
            input_ids=full_ids,
            attention_mask=full_mask,
            pixel_values=None,
            image_hidden_states=ihs_gpu,
        )
    finally:
        actor._action_embeds_buffer[0] = None
        actor._action_insert_pos[0]    = None

    gen_logits = outputs.logits[0, prompt_length - 1 : -1, :]
    loss       = F.cross_entropy(gen_logits, target_ids_gpu[0])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer.step()

    loss_val = loss.item()
    del (input_ids, attention_mask, target_ids_gpu, full_ids, full_mask,
         ihs_gpu, action_embeds, outputs, gen_logits, loss)
    torch.cuda.empty_cache()
    return loss_val


# ─────────────────────────────────────────
# 형식 확인
# ─────────────────────────────────────────

def check_format(actor: ActorModel, env, instruction: str):
    """현재 env 이미지로 형식 출력 확인."""
    try:
        obs   = env.reset()
        image = get_image(obs)
        with torch.no_grad():
            (critique, action_vec, action_ids, *_) = actor.generate(
                image=image, instruction=instruction
            )
        smol_start = len(actor.processor.tokenizer) - 256
        n_action   = sum(1 for t in action_ids if smol_start <= int(t) < smol_start + 256)
        ok = "✓" if n_action == 7 and critique else "✗"
        print(f"  [{ok}] CRITIQUE: {(critique or '없음')[:70]}")
        print(f"      action_tokens={n_action}/7 | {np.round(action_vec, 2)}")
    except Exception as e:
        print(f"  [확인 skip] {e}")


# ─────────────────────────────────────────
# SFT 메인
# ─────────────────────────────────────────

def sft_train(actor: ActorModel):
    print(f"\n{'='*60}")
    print(f"형식 SFT 시작: {NUM_STEPS} 스텝")
    print(f"이미지: LIBERO 실제 이미지 | 태스크: {TASK_IDS}")
    print(f"타겟: CRITIQUE 텍스트 + [ACTION] planner_action [/ACTION]")
    print(f"{'='*60}\n")

    trainable = list(filter(lambda p: p.requires_grad, actor.parameters()))
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable, lr=LEARNING_RATE)
        print("[optimizer] AdamW8bit")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
        print("[optimizer] AdamW")

    # 첫 태스크 환경 준비
    task_idx = 0
    env, instruction = make_env(TASK_IDS[task_idx])
    obs    = env.reset()
    losses = []

    print(f"[태스크 {TASK_IDS[task_idx]}] {instruction}\n")

    for step in range(1, NUM_STEPS + 1):

        # ── 태스크 전환 ──────────────────────────────────
        if step % TASK_SWITCH_EVERY == 1 and step > 1:
            env.close()
            task_idx  = (task_idx + 1) % len(TASK_IDS)
            env, instruction = make_env(TASK_IDS[task_idx])
            obs = env.reset()
            print(f"\n[태스크 전환 → {TASK_IDS[task_idx]}] {instruction}")

        # ── 동일 태스크 내 주기적 reset ──────────────────
        if step % RESET_EVERY == 1 and step > 1:
            obs = env.reset()

        # ── 실제 이미지로 SFT 데이터 생성 ────────────────
        image = get_image(obs)
        try:
            cached_inputs, planner_tokens, target_ids, ihs = make_sft_example(
                actor, image, instruction
            )
        except Exception as e:
            print(f"  [데이터 생성 오류 skip] {e}")
            obs = env.reset()
            continue

        # ── SFT 스텝 ─────────────────────────────────────
        loss_val = sft_step(
            actor, optimizer, cached_inputs, planner_tokens, target_ids, ihs
        )
        losses.append(loss_val)

        # ── 다음 이미지 수집 (플래너 액션으로 env step) ───
        try:
            planner_action = actor.action_tokenizer.bin_indices_to_continuous(
                actor.action_tokenizer.openvla_ids_to_bin_indices(
                    np.array(planner_tokens)
                )
            )
            obs, _, done, _ = env.step(planner_action)
            if done:
                obs = env.reset()
        except Exception:
            obs = env.reset()

        # ── 로그 ─────────────────────────────────────────
        if step % 50 == 0:
            avg = float(np.mean(losses[-50:]))
            print(f"  [Step {step:4d}/{NUM_STEPS}] loss={avg:.4f} "
                  f"| task={TASK_IDS[task_idx]}")

        if step % 100 == 0:
            check_format(actor, env, instruction)

    env.close()

    # 저장
    os.makedirs(SAVE_PATH, exist_ok=True)
    actor.smol.save_pretrained(SAVE_PATH)
    actor.processor.save_pretrained(SAVE_PATH)
    print(f"\nSFT 완료! 저장: {SAVE_PATH}")
    print("다음 단계: smol_train.py (GRPO)")


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main():
    print(f"[CUDA] {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '미설정')}")
    print("Actor 모델 초기화 중... (Planner 서버 필요)")

    actor = ActorModel()
    actor.smol.gradient_checkpointing_enable()

    alloc = torch.cuda.memory_allocated() / 1024**2
    print(f"[VRAM] 모델 로드 후: {alloc:.0f}MB")

    sft_train(actor)
    actor.close()


if __name__ == "__main__":
    main()