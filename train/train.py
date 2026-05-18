"""
GRPO 학습 스크립트
- 환경: LIBERO-Long 3개 태스크, 각 500 에피소드
- TRL GRPOTrainer 대신 직접 GRPO 구현
- LIBERO 환경 reward를 직접 사용

실행 전 준비:
    터미널 1 (openvla_env): python openvla_planner/openvla_inference_code.py
    터미널 2 (qwen_env):   python train/train.py
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
from qwen_actor.actor_model import ActorModel


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

TASK_SUITE    = "libero_long"   # LIBERO-Long (libero_10)
TASK_IDS      = [0, 1, 2]      # 학습할 태스크 3개 (0~9 중 선택)
NUM_EPISODES  = 500             # 태스크당 학습 에피소드 수
MAX_STEPS     = 300             # 에피소드당 최대 스텝 수
IMG_HEIGHT    = 224             # 이미지 높이
IMG_WIDTH     = 224             # 이미지 너비
SAVE_PATH     = "checkpoints"   # 체크포인트 저장 경로
GROUP_SIZE    = 4               # GRPO group sampling 수 (메모리 제한으로 2~4 권장)
LEARNING_RATE = 1e-4
GRAD_ACCUM    = 4               # gradient accumulation steps


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
# 보상 함수
# ─────────────────────────────────────────

def reward_fn(info: dict, done: bool) -> float:
    """
    보상 함수. 여기만 수정하면 보상 설계 변경 가능.

    현재: binary reward (성공 1.0 / 실패 0.0)
    향후: dense reward 추가 가능
        - gripper와 목표물 거리 기반
        - 태스크 진행 단계 기반

    :param info: LIBERO 환경 info dict
    :param done: 에피소드 종료 여부
    :return: reward (float)
    """
    if info.get("success", False):
        return 1.0
    return 0.0


# ─────────────────────────────────────────
# GRPO 핵심 로직
# ─────────────────────────────────────────

def compute_grpo_loss(
    actor: ActorModel,
    image: Image.Image,
    instruction: str,
    env,
    group_size: int = GROUP_SIZE
) -> tuple:
    """
    단일 상태에서 GRPO loss 계산.

    GRPO 동작 방식:
        1. 같은 이미지 + 텍스트로 group_size개 action 샘플링
        2. 각 action을 LIBERO 환경에서 실행해서 reward 수집
        3. Group 내 상대적 advantage 계산
           advantage = (reward - mean) / std
        4. log_prob * advantage로 policy gradient loss 계산

    수정 사항:
        - group sampling 중 env.reset() 제거
          → 같은 상태에서 모든 샘플을 실행해야 GRPO가 유효함
          → 대신 env.set_state()로 초기 상태 저장/복원
        - obs 반환 추가
          → 학습 루프에서 중복 env.step() 방지

    :param actor: ActorModel
    :param image: 현재 환경 이미지
    :param instruction: 태스크 명령
    :param env: LIBERO 환경
    :param group_size: 샘플링할 action 수
    :return: (loss, obs, done, info) 튜플
        - loss: GRPO loss (scalar tensor)
        - obs: 마지막 샘플의 환경 관찰값
        - done: 마지막 샘플의 에피소드 종료 여부
        - info: 마지막 샘플의 환경 info
    """
    rewards   = []
    log_probs = []
    last_obs  = None
    last_done = False
    last_info = {}

    # 현재 환경 상태 저장 (같은 상태에서 group sampling)
    try:
        init_state = env.get_state()
        use_state_restore = True
    except AttributeError:
        # get_state() 지원 안 하면 상태 복원 없이 진행
        use_state_restore = False

    # group_size개 action 샘플링 + reward 수집
    for g in range(group_size):

        # 같은 초기 상태에서 시작 (첫 번째 샘플 제외)
        if g > 0 and use_state_restore:
            env.set_state(init_state)

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
        reward = reward_fn(info, done)
        rewards.append(reward)

        # 마지막 샘플 결과 저장 (학습 루프에서 사용)
        last_obs  = obs
        last_done = done
        last_info = info

    # advantage 계산 (Group 내 상대적 reward)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    if rewards_tensor.std() < 1e-8:
        # 모든 reward가 같으면 advantage = 0 (업데이트 없음)
        advantages = torch.zeros_like(rewards_tensor)
    else:
        advantages = (
            (rewards_tensor - rewards_tensor.mean())
            / (rewards_tensor.std() + 1e-8)
        )

    # GRPO loss 계산
    log_probs_tensor = torch.stack(log_probs)
    loss = -(log_probs_tensor * advantages.to(log_probs_tensor.device)).mean()

    return loss, last_obs, last_done, last_info


# ─────────────────────────────────────────
# 단일 에피소드 실행 (평가용)
# ─────────────────────────────────────────

def run_episode(env, actor: ActorModel, instruction: str) -> dict:
    """
    단일 에피소드 실행 (성능 평가용).
    학습 루프와 별개로 현재 성능 확인에 사용.
    gradient 계산 없음.

    :return: {"reward": float, "success": bool, "steps": int}
    """
    obs     = env.reset()
    success = False

    for step in range(MAX_STEPS):
        image = get_image_from_obs(obs)

        with torch.no_grad():
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

    학습 루프 구조:
        에피소드 시작
        └── 환경 리셋
        └── 스텝 반복
            ├── compute_grpo_loss() 호출
            │   → group_size개 action 샘플링
            │   → LIBERO 실행 → reward 수집
            │   → advantage 계산
            │   → loss 반환 + 마지막 obs 반환
            ├── backward() → optimizer.step()
            └── 반환된 obs로 다음 스텝 진행
                (중복 env.step() 없음)

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

        obs          = env.reset()
        episode_loss = 0.0
        step_count   = 0
        optimizer.zero_grad()

        for step in range(MAX_STEPS):
            image = get_image_from_obs(obs)

            # GRPO loss 계산
            # compute_grpo_loss가 env.step()을 내부에서 실행하고
            # 마지막 obs를 반환하므로 여기서 중복 env.step() 하지 않음
            loss, obs, done, info = compute_grpo_loss(
                actor, image, instruction, env
            )
            episode_loss += loss.item()
            step_count   += 1

            # Gradient accumulation
            (loss / GRAD_ACCUM).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            if done:
                successes.append(info.get("success", False))
                break

        # 마지막 gradient 처리
        if step_count % GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(
                actor.parameters(), max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        losses.append(episode_loss / max(step_count, 1))

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

    # gradient checkpointing으로 VRAM 절약 (필수)
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
