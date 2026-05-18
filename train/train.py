"""
GRPO 학습 스크립트
- 환경: LIBERO-Long 3개 태스크, 각 500 에피소드
- TRL GRPOTrainer 대신 직접 GRPO 구현
- LIBERO 환경 reward를 직접 사용
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../qwen_actor"))

import torch
import numpy as np
from PIL import Image
from torch.optim import AdamW
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from actor_model import ActorModel


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

TASK_SUITE   = "libero_long"
TASK_IDS     = [0, 1, 2]
NUM_EPISODES = 500
MAX_STEPS    = 300
IMG_HEIGHT   = 224
IMG_WIDTH    = 224
SAVE_PATH    = "checkpoints"

# GRPO 설정
GROUP_SIZE   = 4        # 같은 상태에서 샘플링할 action 수 (G)
                        # 메모리 제한으로 2~4 권장
LEARNING_RATE = 1e-4
GRAD_ACCUM    = 4       # gradient accumulation steps


# ─────────────────────────────────────────
# LIBERO 환경 초기화
# ─────────────────────────────────────────

def make_env(task_id: int):
    """
    LIBERO-Long 환경 초기화.

    :param task_id: 태스크 인덱스 (0~9)
    :return: (env, task_name)
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_SUITE]()
    task_name      = task_suite.get_task_name(task_id)
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
    """LIBERO obs → PIL Image"""
    return Image.fromarray(obs["agentview_image"].astype(np.uint8))


# ─────────────────────────────────────────
# GRPO 핵심 로직
# ─────────────────────────────────────────

def compute_grpo_loss(
    actor: ActorModel,
    image: Image.Image,
    instruction: str,
    env,
    group_size: int = GROUP_SIZE
) -> torch.Tensor:
    """
    단일 상태에서 GRPO loss 계산.

    GRPO 동작 방식:
        1. 같은 이미지 + 텍스트로 group_size개 action 샘플링
        2. 각 action을 LIBERO 환경에서 실행해서 reward 수집
        3. Group 내 상대적 advantage 계산
           advantage = (reward - mean) / std
        4. log_prob * advantage로 policy gradient loss 계산

    :param actor: ActorModel
    :param image: 현재 환경 이미지
    :param instruction: 태스크 명령
    :param env: LIBERO 환경
    :param group_size: 샘플링할 action 수
    :return: GRPO loss (scalar tensor)
    """

    rewards   = []
    log_probs = []

    # 1. group_size개 action 샘플링 + reward 수집
    for _ in range(group_size):

        # Actor forward → logits
        logits = actor.forward(image, instruction)

        # action token 샘플링 (temperature sampling)
        action_token_ids = torch.multinomial(
            torch.softmax(logits[0, -7:, :], dim=-1),
            num_samples=1
        ).squeeze(-1)  # (7,)

        # log probability 계산
        log_prob = torch.log_softmax(logits[0, -7:, :], dim=-1)
        token_log_prob = log_prob[
            torch.arange(7), action_token_ids
        ].sum()  # scalar
        log_probs.append(token_log_prob)

        # action vector 복원
        action_vector = actor.action_tokenizer.decode_token_ids_to_actions(
            action_token_ids.cpu().numpy()
        )

        # LIBERO 환경 실행 → reward
        obs, _, done, info = env.step(action_vector)
        reward = 1.0 if info.get("success", False) else 0.0
        rewards.append(reward)

        # 환경 리셋 (다음 샘플링을 위해)
        if done:
            env.reset()

    # 2. advantage 계산 (Group 내 상대적 reward)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    if rewards_tensor.std() < 1e-8:
        # 모든 reward가 같으면 advantage = 0 (업데이트 없음)
        advantages = torch.zeros_like(rewards_tensor)
    else:
        advantages = (
            (rewards_tensor - rewards_tensor.mean())
            / (rewards_tensor.std() + 1e-8)
        )

    # 3. GRPO loss 계산
    log_probs_tensor = torch.stack(log_probs)
    loss = -(log_probs_tensor * advantages.to(log_probs_tensor.device)).mean()

    return loss


# ─────────────────────────────────────────
# 단일 에피소드 실행 (평가용)
# ─────────────────────────────────────────

def run_episode(env, actor: ActorModel, instruction: str) -> dict:
    """
    단일 에피소드 실행 (성능 평가용).
    학습 루프와 별개로 현재 성능 확인에 사용.

    :return: {"reward": float, "success": bool, "steps": int}
    """
    obs     = env.reset()
    success = False

    for step in range(MAX_STEPS):
        image = get_image_from_obs(obs)

        _, action_vector = actor.generate(
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
    """
    단일 태스크 GRPO 학습.
    NUM_EPISODES 에피소드 동안 학습 + 주기적 평가.

    :param actor: ActorModel
    :param task_id: LIBERO-Long 태스크 인덱스
    :return: 학습 결과 통계
    """
    env, task_name = make_env(task_id)
    instruction    = task_name

    print(f"\n{'='*50}")
    print(f"태스크 {task_id}: {task_name}")
    print(f"에피소드: {NUM_EPISODES} | Group size: {GROUP_SIZE}")
    print(f"{'='*50}")

    # Optimizer: LoRA + Projection Layer만 학습
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, actor.parameters()),
        lr=LEARNING_RATE
    )

    losses    = []
    successes = []

    for episode in range(NUM_EPISODES):

        obs = env.reset()
        episode_loss = 0.0

        optimizer.zero_grad()

        for step in range(MAX_STEPS):
            image = get_image_from_obs(obs)

            # GRPO loss 계산
            loss = compute_grpo_loss(
                actor, image, instruction, env
            )
            episode_loss += loss.item()

            # Gradient accumulation
            (loss / GRAD_ACCUM).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            # 다음 스텝
            _, action_vector = actor.generate(image, instruction)
            obs, _, done, info = env.step(action_vector)

            if done:
                successes.append(info.get("success", False))
                break

        losses.append(episode_loss)

        # 로그 출력
        if (episode + 1) % 10 == 0:
            recent_success = np.mean(successes[-10:]) * 100 if successes else 0
            recent_loss    = np.mean(losses[-10:])
            print(
                f"[태스크 {task_id}] "
                f"에피소드 {episode+1}/{NUM_EPISODES} | "
                f"loss: {recent_loss:.4f} | "
                f"성공률: {recent_success:.1f}%"
            )

        # 체크포인트 저장
        if (episode + 1) % 100 == 0:
            save_dir = f"{SAVE_PATH}/task_{task_id}_ep_{episode+1}"
            actor.qwen.save_pretrained(save_dir)
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
    """
    전체 학습 루프.
    태스크 0 → 1 → 2 순서로 500 에피소드씩 학습.
    """
    print("Actor 모델 초기화 중...")
    print("Planner 서버(openvla_inference_code.py)가 실행 중이어야 합니다.")

    actor = ActorModel()

    # gradient checkpointing으로 VRAM 절약
    actor.qwen.gradient_checkpointing_enable()

    all_stats = []

    for task_id in TASK_IDS:
        stats = train_on_task(actor, task_id)
        all_stats.append(stats)

    # 전체 결과
    print(f"\n{'='*50}")
    print("전체 학습 완료!")
    print(f"{'='*50}")
    for stats in all_stats:
        print(
            f"태스크 {stats['task_id']} ({stats['task_name']}): "
            f"성공률 {stats['success_rate']:.1f}%"
        )

    # 최종 모델 저장
    actor.qwen.save_pretrained(f"{SAVE_PATH}/final")
    print(f"최종 모델 저장: {SAVE_PATH}/final")

    actor.close()


if __name__ == "__main__":
    main()