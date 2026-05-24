"""
GRPO 학습 스크립트 (RLinf 방식 + LR 스케줄러 + 풍부한 보상함수)
- 환경: LIBERO-Long 3개 태스크, 각 300 에피소드
- rollout 단계와 학습 단계 완전 분리 (VRAM 효율화)
- Linear Warmup + Cosine Annealing LR 스케줄러
- 다중 보상 신호 적용
- rollout에서 generate() 사용 → critique 텍스트 + action token 생성

OOM 수정 핵심:
    스텝마다 즉시 backward() + optimizer.step()
    → 각 스텝의 gradient graph가 즉시 해제 → VRAM 절약

실행 전 준비:
    터미널 1 (openvla_env): python openvla_planner/openvla_inference_code.py
    터미널 2 (qwen_env):
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        python train/smol_train.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

import torch
import numpy as np
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from SmolVLM_actor.smol_actor_model import ActorModel


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

TASK_SUITE      = "libero_10"
TASK_IDS        = [0, 1, 2]
NUM_EPISODES    = 100
MAX_STEPS       = 15
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
SAVE_PATH       = "checkpoints"
GROUP_SIZE      = 3

# LR 스케줄러 설정
LEARNING_RATE   = 1e-4
LR_MIN          = 1e-6
WARMUP_EPISODES = 20


# ─────────────────────────────────────────
# LIBERO 환경 초기화
# ─────────────────────────────────────────

def make_env(task_id: int):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_SUITE]()
    task_names     = task_suite.get_task_names()
    task_name      = task_names[task_id]
    task_bddl_file = task_suite.get_task_bddl_file_path(task_id)

    print(f"태스크 로드: {task_name}")

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

def reward_fn(action_vector: np.ndarray, obs, info: dict, done: bool) -> float:
    total_reward = 0.0
    try:
        success = info.get("success", False)
        if success:
            total_reward += 1.0

        collision = info.get("collision", False)
        if collision:
            total_reward -= 0.5

        current_step = info.get("step", 0)
        free_steps   = 100
        if current_step > free_steps:
            overtime      = current_step - free_steps
            total_reward -= 0.01 * overtime

        ee_velocity      = info.get("ee_velocity", 0.0)
        stability_reward = max(0.0, 1.0 - abs(ee_velocity))
        total_reward    += 0.3 * stability_reward

        if np.any(np.isnan(action_vector)):
            total_reward -= 1.0

        total_reward = float(np.clip(total_reward, -1.0, 2.0))

    except Exception as e:
        print(f"[reward_fn 오류]: {e}")
        total_reward = -1.0

    return total_reward


# ─────────────────────────────────────────
# LR 스케줄러 생성
# ─────────────────────────────────────────

def make_scheduler(optimizer, num_episodes: int):
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPISODES
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_episodes - WARMUP_EPISODES,
        eta_min=LR_MIN
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPISODES]
    )
    return scheduler


# ─────────────────────────────────────────
# rollout 수집 (RLinf 핵심 - no_grad)
# ─────────────────────────────────────────

def collect_rollout(
    actor: ActorModel,
    env,
    instruction: str,
    group_size: int = GROUP_SIZE
) -> list:
    trajectories = []
    obs = env.reset()

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
                # 🌟 수정: generate()가 반환하는 6개의 값을 모두 받습니다 (new_tokens 추가)
                critique, action_vector, action_token_ids, planner_tokens, cached_inputs, new_tokens = actor.generate(
                    image=image,
                    instruction=instruction
                )

            print(f"  [스텝 {step+1}] critique:      {critique[:60] if critique else '(없음)'}")
            print(f"  [스텝 {step+1}] action_vector: {action_vector}")
            print(f"  [스텝 {step+1}] action_tokens: {action_token_ids}")

            obs, _, done, info = env.step(action_vector)
            reward = reward_fn(action_vector, obs, info, done)

            # trajectory 저장
            group_traj.append({
                "action_token_ids":      torch.tensor(action_token_ids, dtype=torch.long),
                "planner_action_tokens": planner_tokens,
                "cached_inputs":         cached_inputs,
                "new_tokens":            new_tokens,      # 🌟 새로 추가: Actor가 생성한 전체 토큰 텐서
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
# trajectory 기반 GRPO loss 계산 (학습 단계)
# ─────────────────────────────────────────

def compute_grpo_loss_from_trajectories(
    actor: ActorModel,
    trajectories: list,
    optimizer: AdamW
) -> float:
    rewards = torch.tensor(
        [traj[-1]["reward"] for traj in trajectories],
        dtype=torch.float32
    )

    if len(rewards) < 2 or rewards.std() < 1e-8:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss  = 0.0
    total_steps = 0

    for g, traj in enumerate(trajectories):
        adv = advantages[g].to("cuda")

        for step_data in traj:
            optimizer.zero_grad()

            #Actor가 뱉어냈던 전체 문장 텐서를 GPU로 올려 forward()에 전달
            new_tokens = step_data["new_tokens"].to("cuda")
            
            logits, input_ids = actor.forward(
                planner_action_tokens=step_data["planner_action_tokens"],
                cached_inputs=step_data["cached_inputs"],
                new_tokens=new_tokens #새로 추가됨
            )

            #[수학적 붕괴 해결 파트]
            # 전체 input_ids(프롬프트 + 생성된 토큰)에서 프롬프트의 길이를 빼서 오프셋을 맞춤
            prompt_length = input_ids.shape[1] - new_tokens.shape[0]
            
            # Logit의 성질상 인덱스 i는 토큰 i+1을 예측함
            # 따라서 새로 생성된 토큰들(new_tokens)을 예측한 Logits의 정확한 슬라이싱 범위는 [prompt_length - 1 : -1]
            gen_logits = logits[0, prompt_length - 1 : -1, :] 
            
            log_prob = torch.log_softmax(gen_logits, dim=-1)
            
            # 전체 생성된 문장 (Critique 비판 텍스트 + Action tokens 전체) 의 결합 로그 확률 합산
            token_log_prob = log_prob[
                torch.arange(new_tokens.shape[0]), new_tokens
            ].sum()

            #Actor의 전체 사고 과정(Critique)과 행동(Action)에 Advantage 스칼라를 곱해 GRPO Loss 산출
            step_loss = -(token_log_prob * adv)

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += step_loss.item()
            total_steps += 1

            del logits, input_ids, step_loss, new_tokens
            torch.cuda.empty_cache()

    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────
# 단일 에피소드 실행 (평가용)
# ─────────────────────────────────────────

def run_episode(env, actor: ActorModel, instruction: str) -> dict:
    obs     = env.reset()
    success = False

    for step in range(MAX_STEPS):
        image = get_image_from_obs(obs)

        with torch.no_grad():
            #에러 방어: 리턴 개수가 늘어났으므로 *_ (나머지 몽땅)로 유연하게 언팩 처리
            _, action_vector, *_ = actor.generate(
                image=image,
                instruction=instruction
            )

        obs, _, done, info = env.step(action_vector)

        if done:
            success = info.get("success", False)
            break

    return {
        "reward":  1.0 if success else 0.0,
        "success": success,
        "steps":   step + 1
    }


# ─────────────────────────────────────────
# 태스크 학습
# ─────────────────────────────────────────

def train_on_task(actor: ActorModel, task_id: int) -> dict:
    env, task_name = make_env(task_id)
    instruction    = task_name

    print(f"\n{'='*50}")
    print(f"태스크 {task_id}: {task_name}")
    print(f"에피소드: {NUM_EPISODES} | Group size: {GROUP_SIZE} | Max steps: {MAX_STEPS}")
    print(f"LR: {LEARNING_RATE} (warmup {WARMUP_EPISODES}ep) → {LR_MIN} (cosine)")
    print(f"{'='*50}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, actor.parameters()),
        lr=LEARNING_RATE
    )
    scheduler = make_scheduler(optimizer, NUM_EPISODES)

    losses    = []
    successes = []

    for episode in range(NUM_EPISODES):

        print(f"\n[에피소드 {episode+1}] rollout 수집 중...")

        trajectories = collect_rollout(actor, env, instruction, GROUP_SIZE)

        last_done = trajectories[0][-1]["done"]
        if last_done:
            successes.append(trajectories[0][-1]["reward"] >= 1.0)

        loss_val = compute_grpo_loss_from_trajectories(
            actor, trajectories, optimizer
        )

        scheduler.step()
        torch.cuda.empty_cache()

        losses.append(loss_val)
        current_lr = scheduler.get_last_lr()[0]

        if (episode + 1) % 10 == 0:
            recent_success = np.mean(successes[-10:]) * 100 if successes else 0
            recent_loss    = np.mean(losses[-10:])
            print(
                f"[태스크 {task_id}] "
                f"에피소드 {episode+1}/{NUM_EPISODES} | "
                f"loss: {recent_loss:.4f} | "
                f"성공률: {recent_success:.1f}% | "
                f"lr: {current_lr:.2e}"
            )

        if (episode + 1) % 100 == 0:
            save_dir = f"{SAVE_PATH}/task_{task_id}_ep_{episode+1}"
            actor.smol.save_pretrained(save_dir)
            print(f"체크포인트 저장: {save_dir}")

    env.close()

    stats = {
        "task_id":      task_id,
        "task_name":    task_name,
        "avg_loss":     np.mean(losses),
        "success_rate": np.mean(successes) * 100 if successes else 0,
    }

    print(f"\n[태스크 {task_id} 완료]")
    print(f"  평균 loss:   {stats['avg_loss']:.4f}")
    print(f"  최종 성공률: {stats['success_rate']:.1f}%")

    return stats


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main():
    print("Actor 모델 초기화 중...")
    print("Planner 서버(openvla_inference_code.py)가 실행 중이어야 합니다.")

    actor = ActorModel()
    actor.smol.gradient_checkpointing_enable()

    all_stats = []

    for task_id in TASK_IDS:
        stats = train_on_task(actor, task_id)
        all_stats.append(stats)

    print(f"\n{'='*50}")
    print("전체 학습 완료!")
    print(f"{'='*50}")
    for stats in all_stats:
        print(
            f"태스크 {stats['task_id']} ({stats['task_name']}): "
            f"성공률 {stats['success_rate']:.1f}%"
        )

    actor.smol.save_pretrained(f"{SAVE_PATH}/final")
    print(f"최종 모델 저장: {SAVE_PATH}/final")

    actor.close()


if __name__ == "__main__":
    main()