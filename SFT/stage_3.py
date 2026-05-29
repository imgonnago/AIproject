"""
smol_sft_stage3.py — Stage 3: 액션 토큰 추종 학습 (700 스텝)

[목적]
    Stage 2 체크포인트 기반.
    텍스트는 자유 (loss=0), 액션 토큰만 OpenVLA를 따라가도록 학습.
    GRPO 시작점이 OpenVLA 벤치마크 수준으로 보장됨.

[핵심 설계]
    reset_action_lm_head(): Stage 2 로드 후 편향 재제거
    get_scene_text():        Stage 1,2와 동일한 방식
    손실 마스크:             액션 토큰 7개만 loss=1, 텍스트 = loss=0
                             → 텍스트 다양성 보존 + 액션 정확도 향상

로드: checkpoints/sft_stage2
저장: checkpoints/sft_stage3
다음: smol_train.py (GRPO)

실행:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/smol_sft_stage3.py
"""

import os
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from safetensors.torch import load_file

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from SmolVLM_actor.smol_actor_model import ActorModel

NUM_STEPS         = 700
LEARNING_RATE     = 1e-5
TASK_SUITE        = "libero_10"
TASK_IDS          = [0, 1, 2, 3, 4]
TASK_SWITCH_EVERY = 100
RESET_EVERY       = 10
IMG_HEIGHT        = 224
IMG_WIDTH         = 224
LOAD_PATH         = "checkpoints/sft_stage2"
SAVE_PATH         = "checkpoints/sft_stage3"
FALLBACK_TEXT     = "I observe the scene and evaluate the proposed action."


# ─────────────────────────────────────────
# lm_head action token 편향 제거
# ─────────────────────────────────────────

def reset_action_lm_head(actor: ActorModel):
    lm_head = actor.smol.get_output_embeddings()
    regular = lm_head.weight.data[:-256]
    mean = regular.mean(dim=0, keepdim=True)
    std  = regular.std(dim=0, keepdim=True).clamp(min=1e-6)
    with torch.no_grad():
        new_w = torch.normal(mean.expand(256, -1), std.expand(256, -1))
        lm_head.weight.data[-256:] = new_w.to(lm_head.weight.dtype)
    after = lm_head.weight.data[-256:].norm(dim=1).mean().item()
    print(f"[reset_action_lm_head] 완료 | action token norm: {after:.3f}")


# ─────────────────────────────────────────
# 체크포인트 로드
# ─────────────────────────────────────────

def load_checkpoint(actor: ActorModel, path: str):
    print(f"[체크포인트 로드] {path}")
    sf_files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    if not sf_files:
        print("  경고: safetensors 없음, 기본 가중치 사용")
        return
    state_dict = {}
    for f in sf_files:
        state_dict.update(load_file(os.path.join(path, f)))
    missing, unexpected = actor.smol.load_state_dict(state_dict, strict=False)
    print(f"  완료 | missing={len(missing)} unexpected={len(unexpected)}")


# ─────────────────────────────────────────
# 장면 텍스트 생성 (Stage 1에서 확인된 방식)
# ─────────────────────────────────────────

def get_scene_text(actor: ActorModel, image: Image.Image,
                   instruction: str, verbose: bool = False) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Briefly describe the scene for: {instruction}"}
        ]
    }]
    inputs = actor.processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            with actor.smol.disable_adapter():
                out = actor.smol.generate(
                    **inputs, max_new_tokens=40,
                    do_sample=True, temperature=0.7,
                )
        prompt_len = inputs["input_ids"].shape[1]
        text = actor.processor.decode(
            out[0][prompt_len:], skip_special_tokens=True
        ).strip()
        del out
    except Exception as e:
        print(f"  [scene_text ERROR] {e}")
        text = ""

    if verbose:
        print(f"  [scene_text] '{text[:100] if text else '(없음)'}'")

    del inputs
    torch.cuda.empty_cache()
    return text if text else FALLBACK_TEXT


# ─────────────────────────────────────────
# Scene text 사전 수집
# ─────────────────────────────────────────

def precompute_scene_texts(actor: ActorModel, env, instruction: str,
                           n: int = 60) -> list:
    print(f"  [scene_text 사전 수집] {n}개 수집 중...")
    texts = []
    obs = env.reset()
    for i in range(n):
        image = get_image(obs)
        text  = get_scene_text(actor, image, instruction, verbose=(i < 3))
        texts.append(text)
        obs, _, done, _ = env.step(np.zeros(7))
        if done:
            obs = env.reset()
    torch.cuda.empty_cache()
    unique = len(set(texts))
    print(f"  [scene_text 사전 수집] 완료 | 고유 텍스트: {unique}/{n}")
    return texts


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


def get_image(obs):
    return Image.fromarray(obs["agentview_image"].astype(np.uint8))


# ─────────────────────────────────────────
# SFT 스텝 (액션 토큰만 loss=1)
# ─────────────────────────────────────────

def sft_step(actor, optimizer, cached_inputs,
             planner_tokens, target_ids, loss_mask, ihs):
    """
    Stage 3: 액션 토큰 7개만 cross-entropy.
    텍스트 위치는 loss_mask=0 → 텍스트 다양성 보존.
    """
    optimizer.zero_grad()

    if loss_mask.sum() < 1:
        return 0.0

    target_gpu    = target_ids.to("cuda")
    loss_mask_gpu = loss_mask.to("cuda")

    logits, prompt_length = actor.forward(
        cached_inputs=cached_inputs,
        new_tokens=target_gpu[0],
        planner_action_tokens=planner_tokens,
        image_hidden_states=ihs
    )

    gen_logits     = logits[0, prompt_length - 1 : -1]
    per_token_loss = F.cross_entropy(gen_logits, target_gpu[0], reduction="none")
    loss           = (per_token_loss * loss_mask_gpu).sum() / loss_mask_gpu.sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer.step()

    val = loss.item()
    del logits, gen_logits, per_token_loss, loss, target_gpu, loss_mask_gpu
    torch.cuda.empty_cache()
    return val


# ─────────────────────────────────────────
# 확인
# ─────────────────────────────────────────

def check_generation(actor, env, instruction):
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
# 학습 루프
# ─────────────────────────────────────────

def train_stage3(actor: ActorModel):
    print(f"\n{'='*60}")
    print(f"Stage 3: 액션 추종 학습 ({NUM_STEPS} 스텝)")
    print(f"텍스트: 자유 (loss=0) | 액션: OpenVLA 추종 (loss=1)")
    print(f"{'='*60}\n")

    load_checkpoint(actor, LOAD_PATH)
    reset_action_lm_head(actor)

    trainable = list(filter(lambda p: p.requires_grad, actor.parameters()))
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable, lr=LEARNING_RATE)
        print("[optimizer] AdamW8bit")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    smol_start = len(actor.processor.tokenizer) - 256

    task_idx = 0
    env, instruction = make_env(TASK_IDS[task_idx])
    obs = env.reset()
    print(f"[태스크 {TASK_IDS[task_idx]}] {instruction[:60]}\n")

    scene_texts = precompute_scene_texts(actor, env, instruction)
    obs = env.reset()

    losses = []

    for step in range(1, NUM_STEPS + 1):
        if step % TASK_SWITCH_EVERY == 1 and step > 1:
            env.close()
            task_idx = (task_idx + 1) % len(TASK_IDS)
            env, instruction = make_env(TASK_IDS[task_idx])
            obs = env.reset()
            print(f"\n[태스크 전환 → {TASK_IDS[task_idx]}] {instruction[:60]}")
            scene_texts = precompute_scene_texts(actor, env, instruction)
            obs = env.reset()

        if step % RESET_EVERY == 1 and step > 1:
            obs = env.reset()

        image = get_image(obs)

        with torch.no_grad():
            (_, _, _, planner_tokens, cached_inputs, _, _, ihs) = actor.generate(
                image=image, instruction=instruction
            )

        # 사전 수집된 텍스트에서 랜덤 선택
        scene_text = scene_texts[np.random.randint(len(scene_texts))]

        planner_bin = actor.action_tokenizer.openvla_ids_to_bin_indices(
            np.array(planner_tokens)
        )
        action_str  = " ".join([f"<action_{b}>" for b in planner_bin])
        eos         = actor.processor.tokenizer.eos_token
        target_text = f"{scene_text}\n\n[ACTION] {action_str} [/ACTION]{eos}"

        target_ids = actor.processor.tokenizer(
            target_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"]

        # 액션 토큰 위치만 loss=1
        target_np  = target_ids.numpy()[0]
        is_action  = (target_np >= smol_start) & (target_np < smol_start + 256)
        loss_mask  = torch.FloatTensor(is_action.astype(np.float32))

        loss_val = sft_step(
            actor, optimizer, cached_inputs,
            planner_tokens, target_ids, loss_mask, ihs
        )
        losses.append(loss_val)

        try:
            planner_action = actor.action_tokenizer.bin_indices_to_continuous(
                actor.action_tokenizer.openvla_ids_to_bin_indices(np.array(planner_tokens))
            )
            obs, _, done, _ = env.step(planner_action)
            if done:
                obs = env.reset()
        except Exception:
            obs = env.reset()

        if step % 50 == 0:
            valid = [l for l in losses[-50:] if l > 0]
            avg   = float(np.mean(valid)) if valid else 0.0
            print(f"  [Step {step:4d}/{NUM_STEPS}] action_loss={avg:.4f} | task={TASK_IDS[task_idx]}")
        if step % 100 == 0:
            check_generation(actor, env, instruction)

    env.close()
    os.makedirs(SAVE_PATH, exist_ok=True)
    actor.smol.save_pretrained(SAVE_PATH)
    actor.processor.save_pretrained(SAVE_PATH)
    print(f"\n[Stage 3 완료] 저장: {SAVE_PATH}")
    print("다음: smol_train.py (GRPO) — checkpoint 경로를 'checkpoints/sft_stage3'으로 설정")


def main():
    print(f"[CUDA] {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '미설정')}")
    actor = ActorModel()
    actor.smol.gradient_checkpointing_enable()
    train_stage3(actor)
    actor.close()


if __name__ == "__main__":
    main()