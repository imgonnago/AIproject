"""
GRPO 학습 스크립트

[변경사항]
    generate() 반환값: 8-tuple → 7-tuple (image_hidden_states 제거)
    forward() 시그니처: planner_action_tokens 유지, image_hidden_states 제거
    GRPO loss 가중치: token ID 기반 (<action_N> 토큰 여부로 구분)

실행:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/smol_train.py
"""

import sys
import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256"
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

import re
import torch
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from SmolVLM_actor.smol_actor_model import ActorModel


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────

TASK_SUITE      = "libero_spatial"
TASK_IDS        = [2]
NUM_EPISODES    = 60
MAX_STEPS       = 50
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
SAVE_PATH       = "checkpoints/smol_grpo"

LEARNING_RATE   = 2e-4
LR_MIN          = 1e-6
WARMUP_EPISODES = 5

# token ID 기반 loss 가중치
# <action_N> 토큰 (smol_action_start~): ACTION_WEIGHT
# 일반 텍스트 토큰 (critique 등):       CRITIQUE_WEIGHT
CRITIQUE_WEIGHT = 0.3
ACTION_WEIGHT   = 0.7


# ─────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────

def make_optimizer(actor: ActorModel) -> torch.optim.Optimizer:
    trainable = list(filter(lambda p: p.requires_grad, actor.parameters()))
    count = sum(p.numel() for p in trainable)
    print(f"[optimizer] 학습 파라미터: {count:,}")
    if actor.vram_cfg["use_8bit_adam"]:
        try:
            from bitsandbytes.optim import AdamW8bit
            print("[optimizer] AdamW8bit")
            return AdamW8bit(trainable, lr=LEARNING_RATE)
        except ImportError:
            pass
    print("[optimizer] 표준 AdamW")
    return torch.optim.AdamW(trainable, lr=LEARNING_RATE)


# ─────────────────────────────────────────
# LIBERO 환경
# ─────────────────────────────────────────

def make_env(task_id: int):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_SUITE]()
    task_name      = task_suite.get_task_names()[task_id]
    task_bddl_file = task_suite.get_task_bddl_file_path(task_id)
    print(f"태스크: {task_name}")
    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_bddl_file,
        "camera_heights": IMG_HEIGHT,
        "camera_widths":  IMG_WIDTH,
    })
    env.seed(42)
    return env, task_name

def get_image_from_obs(obs: dict) -> Image.Image:
    return Image.fromarray(obs["agentview_image"].astype(np.uint8))


# ─────────────────────────────────────────
# 보상 함수
# ─────────────────────────────────────────

def _has_real_text(critique: str) -> bool:
    """action 토큰 제거 후 의미있는 단어가 2개 이상인지 확인."""
    if not critique:
        return False
    cleaned = re.sub(r"<action_\d+>", "", critique).strip()
    words   = [w for w in re.split(r"[\s\[\]/]", cleaned) if len(w) >= 2]
    return len(words) >= 2


def reward_fn(
    action_vector: np.ndarray,
    info: dict,
    critique: str = "",
    planner_action: np.ndarray = None
) -> float:
    """
    보상 함수.

    패널티/보상 항목:
        파싱 실패              → -1.0 (즉시 반환)
        태스크 성공            → +1.0
        범위 초과              → -0.1 × 초과 차원 수
        크기 초과 (norm > 2)   → -0.2 × 초과량
        critique 텍스트 없음   → -0.3
        planner 유지 (dev<0.3) → +0.1  (보수적 행동 장려, 두 그룹 간 차이 생성)
        planner 이탈 (dev>1.0) → -0.1  (불필요한 수정 억제)
    """
    try:
        if info.get("parsing_failed", False):
            print(f"  [reward] 파싱 실패 → -1.0")
            return -1.0

        reward, reward_parts = 0.0, []

        if info.get("success", False):
            reward += 1.0
            reward_parts.append("성공(+1.0)")

        if np.any(np.isnan(action_vector)):
            print(f"  [reward] NaN 감지 → -1.0")
            return -1.0

        out_of_range = int(np.sum(np.abs(action_vector) > 1.0))
        if out_of_range > 0:
            p = 0.1 * out_of_range
            reward -= p
            reward_parts.append(f"범위초과(-{p:.1f})")

        mag = float(np.linalg.norm(action_vector[:6]))
        if mag > 2.0:
            p = 0.2 * (mag - 2.0)
            reward -= p
            reward_parts.append(f"크기초과(-{p:.2f})")

        # ── critique 텍스트 없으면 패널티 ──────────────────────────
        if not _has_real_text(critique):
            reward -= 0.3
            reward_parts.append("critique없음(-0.3)")
        else:
            reward = 0.2  # critique 텍스트 있으면 기본 보상 +0.2
            reward_parts.append("critique있음(+0.2)")

        # ── 플래너 편차 보상: 두 그룹 간 자연스러운 차이 생성 ────────
        # deviation < 0.3: 플래너 거의 그대로 → 보수적 행동 보상
        # deviation > 1.0: 크게 수정했는데 성공 안 함 → 억제
        if planner_action is not None:
            deviation = float(np.linalg.norm(action_vector - planner_action))
            success   = info.get("success", False)
            if deviation < 0.3:
                reward += 0.2
                reward_parts.append("planner유지(+0.2)")
            elif deviation > 1.0 and not success:
                reward -= 0.1
                reward_parts.append("planner이탈(-0.1)")
        # ────────────────────────────────────────────────────────────

        final = float(np.clip(reward, -1.0, 2.0))
        print(f"  [reward] {', '.join(reward_parts) or '없음'} → {final:.3f}")
        return final

    except Exception as e:
        print(f"  [reward_fn 오류] {e}")
        return -1.0


# ─────────────────────────────────────────
# LR 스케줄러
# ─────────────────────────────────────────

def make_scheduler(optimizer, num_episodes: int):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPISODES)
    cosine = CosineAnnealingLR(optimizer, T_max=num_episodes - WARMUP_EPISODES, eta_min=LR_MIN)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPISODES])


# ─────────────────────────────────────────
# VRAM 모니터링
# ─────────────────────────────────────────

def log_vram(tag: str = ""):
    alloc  = torch.cuda.memory_allocated()     / 1024**2
    reserv = torch.cuda.memory_reserved()      / 1024**2
    peak   = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  [VRAM{' '+tag if tag else ''}] "
          f"alloc={alloc:.0f}MB reserved={reserv:.0f}MB peak={peak:.0f}MB")


# ─────────────────────────────────────────
# rollout 수집
# ─────────────────────────────────────────

def collect_rollout(actor: ActorModel, env, instruction: str) -> list:
    group_size   = actor.vram_cfg["group_size"]
    trajectories = []
    obs          = env.reset()

    try:
        init_state        = env.get_state()
        use_state_restore = True
    except AttributeError:
        use_state_restore = False

    for g in range(group_size):
        if g > 0 and use_state_restore:
            env.set_state(init_state)

        group_traj = []

        for step in range(MAX_STEPS):
            image = get_image_from_obs(obs)

            with torch.no_grad():
                # generate()는 7-tuple 반환 (image_hidden_states 없음)
                (critique, action_vector, action_token_ids,
                 planner_tokens, cached_inputs,
                 new_tokens, parsing_failed, image_hidden_states) = actor.generate(
                    image=image, instruction=instruction
                )

            print(f"  [G{g+1} S{step+1}] {critique if critique else '(없음)'}")
            print(f"  [G{g+1} S{step+1}] action: {np.round(action_vector, 3)}")
            print("=" * 50)

            obs, _, done, info = env.step(action_vector)

            # planner 연속값 계산 (편차 보상용)
            planner_action = actor.action_tokenizer.bin_indices_to_continuous(
                actor.action_tokenizer.openvla_ids_to_bin_indices(np.array(planner_tokens))
            )

            info["parsing_failed"] = parsing_failed
            reward = reward_fn(
                action_vector, info,
                critique=critique,
                planner_action=planner_action
            )

            group_traj.append({
                "cached_inputs":         cached_inputs,
                "new_tokens":            new_tokens,           # CPU tensor
                "image_hidden_states":   image_hidden_states,  # connector 캡처 (CPU)
                "planner_action_tokens": planner_tokens,       # raw OpenVLA IDs
                "action_token_ids":      torch.tensor(action_token_ids, dtype=torch.long),
                "action_vector":         action_vector,
                "critique":              critique,
                "reward":                reward,
                "done":                  done
            })

            if done:
                break

        trajectories.append(group_traj)

    return trajectories


# ─────────────────────────────────────────
# GRPO loss
# ─────────────────────────────────────────

def compute_grpo_loss_from_trajectories(
    actor: ActorModel,
    trajectories: list,
    optimizer: torch.optim.Optimizer
) -> float:
    """
    token ID 기반 가중치:
        <action_N> 토큰 (smol_action_start~): ACTION_WEIGHT (0.7)
        critique 텍스트 토큰:                 CRITIQUE_WEIGHT (0.3)
    """
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    log_vram("학습 시작 전")

    rewards = torch.tensor(
        [traj[-1]["reward"] for traj in trajectories], dtype=torch.float32
    )

    if len(rewards) < 2 or rewards.std() < 1e-8:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    smol_action_start = len(actor.processor.tokenizer) - 256

    total_loss, total_steps = 0.0, 0

    for g, traj in enumerate(trajectories):
        adv = advantages[g].to("cuda")

        for step_data in traj:
            optimizer.zero_grad()

            new_tokens = step_data["new_tokens"].to("cuda")  # (N,)
            N = new_tokens.shape[0]

            # forward: planner_action_tokens 전달 (action token IDs 구성용)
            logits, prompt_length = actor.forward(
                cached_inputs=step_data["cached_inputs"],
                new_tokens=new_tokens,
                planner_action_tokens=step_data["planner_action_tokens"],
                image_hidden_states=step_data["image_hidden_states"]
            )

            # logit 슬라이싱
            gen_logits = logits[0, prompt_length - 1 : -1, :]  # (N, vocab)
            log_prob   = torch.log_softmax(gen_logits, dim=-1)

            per_token_logp = log_prob[
                torch.arange(N, device="cuda"), new_tokens
            ]  # (N,)

            # token ID 기반 가중치
            is_action = (
                (new_tokens >= smol_action_start) &
                (new_tokens < smol_action_start + 256)
            ).float()

            weights = torch.where(
                is_action.bool(),
                torch.full_like(is_action, ACTION_WEIGHT),
                torch.full_like(is_action, CRITIQUE_WEIGHT)
            )

            weighted_logp = (per_token_logp * weights).sum() / N
            step_loss = -(weighted_logp * adv)

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += step_loss.item()
            total_steps += 1

            if total_steps % 10 == 0:
                n_action = int(is_action.sum().item())
                print(
                    f"  [loss] {step_loss.item():.4f} | "
                    f"adv={adv.item():.3f} | "
                    f"action_tokens={n_action}/7"
                )

            del logits, gen_logits, log_prob, per_token_logp, weights, is_action, step_loss, new_tokens
            torch.cuda.empty_cache()

    log_vram("학습 종료 후")
    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────
# 평가
# ─────────────────────────────────────────

def run_episode(env, actor: ActorModel, instruction: str) -> dict:
    obs, success = env.reset(), False
    for step in range(MAX_STEPS):
        with torch.no_grad():
            _, action_vector, *_ = actor.generate(
                image=get_image_from_obs(obs), instruction=instruction
            )
        obs, _, done, info = env.step(action_vector)
        if done:
            success = info.get("success", False)
            break
    return {"success": success, "steps": step + 1}


# ─────────────────────────────────────────
# 태스크 학습
# ─────────────────────────────────────────

def train_on_task(actor: ActorModel, task_id: int) -> dict:
    env, task_name = make_env(task_id)
    instruction    = task_name

    print(f"\n{'='*60}")
    print(f"태스크 {task_id}: {task_name}")
    print(f"에피소드: {NUM_EPISODES} | GROUP: {actor.vram_cfg['group_size']} | MAX_STEPS: {MAX_STEPS}")
    print(f"{'='*60}")

    optimizer = make_optimizer(actor)
    scheduler = make_scheduler(optimizer, NUM_EPISODES)
    losses, successes = [], []

    for episode in range(NUM_EPISODES):
        print(f"\n[에피소드 {episode+1}/{NUM_EPISODES}]")

        if (episode + 1) % 10 == 0:
            alloc  = torch.cuda.memory_allocated() / 1024**2
            reserv = torch.cuda.memory_reserved()  / 1024**2
            print(f"  [VRAM] allocated={alloc:.0f}MB reserved={reserv:.0f}MB")

        trajectories = collect_rollout(actor, env, instruction)

        if trajectories[0][-1]["done"]:
            successes.append(trajectories[0][-1]["reward"] >= 1.0)

        loss_val = compute_grpo_loss_from_trajectories(actor, trajectories, optimizer)
        scheduler.step()
        torch.cuda.empty_cache()

        losses.append(loss_val)
        lr = scheduler.get_last_lr()[0]

        if (episode + 1) % 10 == 0:
            sr = np.mean(successes[-10:]) * 100 if successes else 0.0
            print(f"  loss={np.mean(losses[-10:]):.4f} | 성공률={sr:.1f}% | lr={lr:.2e}")

        if (episode + 1) % 50 == 0:
            save_dir = f"{SAVE_PATH}/task_{task_id}_ep_{episode+1}"
            actor.smol.save_pretrained(save_dir)
            print(f"  체크포인트 저장: {save_dir}")

    env.close()

    stats = {
        "task_id":      task_id,
        "task_name":    task_name,
        "avg_loss":     float(np.mean(losses)),
        "success_rate": float(np.mean(successes) * 100) if successes else 0.0,
    }
    print(f"\n[태스크 {task_id} 완료] loss={stats['avg_loss']:.4f} | 성공률={stats['success_rate']:.1f}%")
    return stats


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main():
    print(f"[CUDA 설정] {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '미설정')}")
    print("Actor 모델 초기화 중...")
    print("Planner 서버(openvla_inference_code.py)가 실행 중이어야 합니다.")

    actor = ActorModel()
    actor.smol.gradient_checkpointing_enable()
    actor.load_and_scale_checkpoint("checkpoints/sft_stage2_5")
    log_vram("모델 로드 후")

    all_stats = []
    for task_id in TASK_IDS:
        stats = train_on_task(actor, task_id)
        all_stats.append(stats)

    print(f"\n{'='*60}")
    print("전체 학습 완료!")
    for s in all_stats:
        print(f"  태스크 {s['task_id']}: 성공률 {s['success_rate']:.1f}%")

    actor.smol.save_pretrained(f"{SAVE_PATH}/final")
    print(f"최종 모델 저장: {SAVE_PATH}/final")
    actor.close()

if __name__ == "__main__":
    main()