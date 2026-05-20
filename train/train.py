"""
GRPO 학습 스크립트 (RLinf 방식 + LR 스케줄러 적용)
- 환경: LIBERO-Long 3개 태스크, 각 300 에피소드
- rollout 단계와 학습 단계 완전 분리 (VRAM 효율화 핵심)
- Linear Warmup + Cosine Annealing LR 스케줄러 적용
- LIBERO 환경 reward를 직접 사용

RLinf 방식 핵심:
    기존: inference + gradient 동시 실행 → VRAM 항상 최대
    개선: rollout(no_grad) → VRAM 해제 → 학습(gradient) → VRAM 절약

실행 전 준비:
    터미널 1 (openvla_env): python openvla_planner/openvla_inference_code.py
    터미널 2 (qwen_env):
        MUJOCO_GL=osmesa
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        python train/train.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../qwen_actor"))

import torch
import numpy as np
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from qwen_actor.actor_model import ActorModel


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

TASK_SUITE    = "libero_10"
TASK_IDS      = [0, 1, 2]
NUM_EPISODES  = 300             # 태스크당 학습 에피소드 수
MAX_STEPS     = 50              # 에피소드당 최대 스텝 수
IMG_HEIGHT    = 224
IMG_WIDTH     = 224
SAVE_PATH     = "checkpoints"
GROUP_SIZE    = 2               # 같은 상태에서 샘플링할 trajectory 수

# LR 스케줄러 설정
LEARNING_RATE = 1e-4            # 최대 lr (warmup 후 도달)
LR_MIN        = 1e-6            # Cosine Annealing 최소 lr
WARMUP_EPISODES = 20            # warmup 에피소드 수
                                # warmup 동안 lr: 1e-5 → 1e-4로 선형 증가
                                # 이후 cosine annealing으로 1e-6까지 감소


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
# LR 스케줄러 생성
# ─────────────────────────────────────────

def make_scheduler(optimizer, num_episodes: int):
    """
    Linear Warmup + Cosine Annealing 스케줄러 생성.

    학습 단계:
        1. Warmup (WARMUP_EPISODES 동안)
           lr: LEARNING_RATE * 0.1 → LEARNING_RATE 선형 증가
           초반 학습 불안정 방지 (GRPO는 초반 reward sparse해서 중요)

        2. Cosine Annealing (나머지 에피소드 동안)
           lr: LEARNING_RATE → LR_MIN 코사인 감소
           학습 후반부에 미세 조정

    :param optimizer: AdamW optimizer
    :param num_episodes: 전체 에피소드 수
    :return: SequentialLR 스케줄러
    """
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,               # 초기 lr = LEARNING_RATE * 0.1
        end_factor=1.0,                 # 최종 lr = LEARNING_RATE
        total_iters=WARMUP_EPISODES
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_episodes - WARMUP_EPISODES,  # warmup 이후 에피소드 수
        eta_min=LR_MIN
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPISODES]    # warmup 종료 시점
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
    """
    RLinf 방식의 rollout 수집.
    gradient 없이 inference만 수행해서 trajectory 수집.

    기존 방식과의 차이:
        기존: inference + gradient 동시 실행 → VRAM 최대 사용
        개선: no_grad로 inference만 → VRAM 절약
              수집된 trajectory는 학습 단계에서 사용

    같은 초기 상태에서 group_size개 trajectory를 수집해서
    GRPO의 group sampling을 구현함.

    :param actor: ActorModel
    :param env: LIBERO 환경
    :param instruction: 태스크 명령
    :param group_size: 수집할 trajectory 수
    :return: trajectory list (group_size개 에피소드)
        각 에피소드: [{image, instruction, action_token_ids, reward, done}, ...]
    """
    trajectories = []

    # 초기 상태 저장 (같은 상태에서 group_size번 샘플링)
    obs = env.reset()
    try:
        init_state        = env.get_state()
        use_state_restore = True
    except AttributeError:
        use_state_restore = False

    for g in range(group_size):

        # 같은 초기 상태에서 시작 (첫 번째 제외)
        if g > 0 and use_state_restore:
            env.set_state(init_state)

        group_traj = []

        for step in range(MAX_STEPS):
            image = get_image_from_obs(obs)

            # ── inference only (no_grad) ──────────────
            # gradient 없이 action token 샘플링
            # VRAM: activation 저장 안 함 → 절약
            with torch.no_grad():
                logits = actor.forward(image, instruction)

                action_token_ids = torch.multinomial(
                    torch.softmax(logits[0, -7:, :], dim=-1),
                    num_samples=1
                ).squeeze(-1)  # (7,)

                # actor_action_tokenizer의 decode_token_ids_to_actions 사용
                action_vector = actor.action_tokenizer.decode_token_ids_to_actions(
                    action_token_ids.cpu().numpy()
                )

                del logits
                torch.cuda.empty_cache()

            # 환경 실행
            obs, _, done, info = env.step(action_vector)
            reward = reward_fn(info, done)

            group_traj.append({
                "image":            image,
                "instruction":      instruction,
                "action_token_ids": action_token_ids.cpu(),  # CPU로 이동해서 저장
                "reward":           reward,
                "done":             done
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
    trajectories: list
) -> torch.Tensor:
    """
    수집된 trajectory로 GRPO loss 계산.

    RLinf 방식 핵심:
        rollout VRAM이 해제된 후 실행
        → inference VRAM 없이 gradient만 관리
        → VRAM 사용량 대폭 감소

    GRPO 동작:
        1. 각 trajectory의 마지막 reward 수집
        2. group 내 상대적 advantage 계산
           advantage = (reward - mean) / std
        3. 각 스텝의 log_prob 재계산 (gradient 있음)
        4. loss = -(log_prob * advantage).mean()

    :param actor: ActorModel
    :param trajectories: collect_rollout()의 반환값
    :return: GRPO loss (scalar tensor)
    """
    # 각 trajectory의 마지막 reward (에피소드 결과)
    rewards = torch.tensor(
        [traj[-1]["reward"] for traj in trajectories],
        dtype=torch.float32
    )

    # advantage 계산 (group 내 상대적 reward)
    if rewards.std() < 1e-8:
        # 모든 reward가 같으면 advantage = 0 (업데이트 없음)
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss  = torch.tensor(0.0, requires_grad=True).to("cuda")
    total_steps = 0

    for g, traj in enumerate(trajectories):
        adv = advantages[g].to("cuda")

        for step_data in traj:
            # ── 학습 단계: gradient 있음 ──────────────
            # rollout에서 저장한 image, instruction으로
            # log_prob 재계산 (이번엔 gradient 포함)
            logits = actor.forward(
                step_data["image"],
                step_data["instruction"]
            )

            # rollout에서 선택한 action token ID로 log_prob 계산
            action_token_ids = step_data["action_token_ids"].to("cuda")
            log_prob = torch.log_softmax(logits[0, -7:, :], dim=-1)
            token_log_prob = log_prob[
                torch.arange(7), action_token_ids
            ].sum()

            total_loss  = total_loss + (-(token_log_prob * adv))
            total_steps += 1

            del logits
            torch.cuda.empty_cache()

    loss = total_loss / max(total_steps, 1)
    return loss


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
    단일 태스크 GRPO 학습 (RLinf 방식 + LR 스케줄러).

    학습 루프 구조:
        에피소드 시작
        └── [1단계] rollout 수집 (no_grad)
            → group_size개 trajectory 수집
            → VRAM: inference activation 없음
        └── [2단계] GRPO loss 계산 + 업데이트 (gradient)
            → 수집된 trajectory로 log_prob 재계산
            → backward() → optimizer.step() → scheduler.step()
            → rollout VRAM 해제 후 실행 → VRAM 절약

    :param actor: ActorModel
    :param task_id: LIBERO-Long 태스크 인덱스
    :return: 학습 결과 통계
    """
    env, task_name = make_env(task_id)
    instruction    = task_name

    print(f"\n{'='*50}")
    print(f"태스크 {task_id}: {task_name}")
    print(f"에피소드: {NUM_EPISODES} | Group size: {GROUP_SIZE}")
    print(f"LR: {LEARNING_RATE} (warmup {WARMUP_EPISODES}ep) → {LR_MIN} (cosine)")
    print(f"{'='*50}")

    # Optimizer: LoRA + Projection Layer만 학습
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, actor.parameters()),
        lr=LEARNING_RATE
    )

    # LR 스케줄러: Linear Warmup + Cosine Annealing
    scheduler = make_scheduler(optimizer, NUM_EPISODES)

    losses    = []
    successes = []

    for episode in range(NUM_EPISODES):

        # ── [1단계] rollout 수집 (no_grad) ───────────
        # inference만 수행, gradient 없음 → VRAM 절약
        trajectories = collect_rollout(
            actor, env, instruction, GROUP_SIZE
        )

        # rollout 결과에서 성공 여부 확인
        last_done = trajectories[0][-1]["done"]
        if last_done:
            successes.append(trajectories[0][-1]["reward"] == 1.0)

        # ── [2단계] GRPO loss 계산 + 업데이트 ─────────
        # rollout VRAM 해제 후 gradient 계산
        optimizer.zero_grad()

        loss = compute_grpo_loss_from_trajectories(actor, trajectories)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()

        # LR 스케줄러 업데이트 (에피소드마다)
        scheduler.step()

        torch.cuda.empty_cache()

        losses.append(loss.item())

        # ── 매 에피소드 출력 확인 ────────────────────
        current_lr = scheduler.get_last_lr()[0]
        with torch.no_grad():
            sample_image         = get_image_from_obs(env.reset())
            critique, action_vec = actor.generate(
                image=sample_image,
                instruction=instruction
            )
            print(f"\n[에피소드 {episode+1} 출력]")
            print(f"  critique:      {critique if critique else '(없음)'}")
            print(f"  action_vector: {action_vec}")

        # ── 로그 출력 ────────────────────────────────
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

        # ── 체크포인트 저장 ───────────────────────────
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
    태스크 0 → 1 → 2 순서로 300 에피소드씩 학습.
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