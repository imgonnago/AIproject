"""
GRPO 학습 스크립트 (RLinf 방식 + LR 스케줄러 + 풍부한 보상함수)
- 환경: LIBERO-Long 3개 태스크, 각 300 에피소드
- rollout 단계와 학습 단계 완전 분리 (VRAM 효율화)
- Linear Warmup + Cosine Annealing LR 스케줄러
- 다중 보상 신호 적용

실행 전 준비:
    터미널 1 (openvla_env): python openvla_planner/openvla_inference_code.py
    터미널 2 (qwen_env):
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        python train/train.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../qwen_actor"))
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
NUM_EPISODES    = 300
MAX_STEPS       = 20              # VRAM 절약을 위해 줄임
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
SAVE_PATH       = "checkpoints"
GROUP_SIZE      = 1               # VRAM 절약을 위해 줄임

# LR 스케줄러 설정
LEARNING_RATE   = 1e-4
LR_MIN          = 1e-6
WARMUP_EPISODES = 20


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

def reward_fn(action_vector: np.ndarray, obs, info: dict, done: bool) -> float:
    """
    다중 보상 신호를 결합한 보상 함수.

    보상 구성:
        +1.0  태스크 성공
        -0.5  충돌 발생
        -0.01 × overtime  MAX_STEPS 초과 시 시간 패널티 (100 스텝 이후부터)
        +0.3  × stability  end-effector 속도 안정성 보상
        +0.05 critique 텍스트 생성 시 보상 (← 이 함수에선 미적용, generate() 단계에서 처리)
        -1.0  action_vector에 NaN 포함 시 패널티
        clip  최종 보상을 [-1.0, 2.0]으로 클리핑

    :param action_vector: 실행된 action vector (7,)
    :param obs: 환경 관찰값
    :param info: 환경 info dict
    :param done: 에피소드 종료 여부
    :return: 보상값 (float)
    """
    total_reward = 0.0

    try:
        # 1. 태스크 성공 보상
        success = info.get("success", False)
        if success:
            total_reward += 1.0

        # 2. 충돌 패널티
        collision = info.get("collision", False)
        if collision:
            total_reward -= 0.5

        # 3. 시간 패널티 (100 스텝 이후부터)
        current_step = info.get("step", 0)
        free_steps   = 100
        if current_step > free_steps:
            overtime      = current_step - free_steps
            total_reward -= 0.01 * overtime

        # 4. end-effector 안정성 보상
        ee_velocity      = info.get("ee_velocity", 0.0)
        stability_reward = max(0.0, 1.0 - abs(ee_velocity))
        total_reward    += 0.3 * stability_reward

        # 5. NaN 패널티
        if np.any(np.isnan(action_vector)):
            total_reward -= 1.0

        # 6. 최종 클리핑
        total_reward = float(np.clip(total_reward, -1.0, 2.0))

    except Exception as e:
        print(f"[reward_fn 오류]: {e}")
        total_reward = -1.0

    return total_reward


# ─────────────────────────────────────────
# LR 스케줄러 생성
# ─────────────────────────────────────────

def make_scheduler(optimizer, num_episodes: int):
    """
    Linear Warmup + Cosine Annealing 스케줄러 생성.

    학습 단계:
        1. Warmup (WARMUP_EPISODES 동안)
           lr: LEARNING_RATE × 0.1 → LEARNING_RATE 선형 증가
           초반 불안정 방지 (GRPO는 초반 reward sparse해서 중요)

        2. Cosine Annealing (나머지 에피소드 동안)
           lr: LEARNING_RATE → LR_MIN 코사인 감소
           학습 후반 미세 조정

    :param optimizer: AdamW optimizer
    :param num_episodes: 전체 에피소드 수
    :return: SequentialLR 스케줄러
    """
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
    """
    RLinf 방식의 rollout 수집.
    gradient 없이 inference만 수행해서 trajectory 수집.

    기존 방식과의 차이:
        기존: inference + gradient 동시 실행 → VRAM 최대
        개선: no_grad inference만 → VRAM 절약
              수집된 trajectory는 학습 단계에서 사용

    :param actor: ActorModel
    :param env: LIBERO 환경
    :param instruction: 태스크 명령
    :param group_size: 수집할 trajectory 수
    :return: group_size개 trajectory list
    """
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

            # inference only (no_grad) → VRAM 절약
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

            # 보상 함수 적용
            reward = reward_fn(action_vector, obs, info, done)

            # critique 보상 (+0.05): generate()로 텍스트 확인
            # 매 스텝마다 generate() 호출은 너무 느리므로
            # rollout 중에는 action_vector 기반 보상만 사용

            group_traj.append({
                "image":            image,
                "instruction":      instruction,
                "action_token_ids": action_token_ids.cpu(),
                "action_vector":    action_vector,
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
        rollout VRAM 해제 후 실행
        → inference VRAM 없이 gradient만 관리 → VRAM 감소

    OOM 수정:
        기존: total_loss에 직접 누적 → gradient graph 무한 누적
        수정: losses_list에 개별 저장 → stack().mean()으로 한 번에 계산

    GRPO 동작:
        1. 각 trajectory의 마지막 reward 수집
        2. group 내 상대적 advantage 계산
        3. log_prob 재계산 (gradient 있음)
        4. loss = -(log_prob × advantage).mean()

    :param actor: ActorModel
    :param trajectories: collect_rollout() 반환값
    :return: GRPO loss (scalar tensor)
    """
    # 각 trajectory의 마지막 reward (에피소드 결과)
    rewards = torch.tensor(
        [traj[-1]["reward"] for traj in trajectories],
        dtype=torch.float32
    )

    # advantage 계산
    if rewards.std() < 1e-8:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    # OOM 수정: list에 개별 저장 후 한 번에 stack
    losses_list = []

    for g, traj in enumerate(trajectories):
        adv = advantages[g].to("cuda")

        for step_data in traj:
            # 학습 단계: gradient 있음
            logits = actor.forward(
                step_data["image"],
                step_data["instruction"]
            )

            action_token_ids = step_data["action_token_ids"].to("cuda")
            log_prob = torch.log_softmax(logits[0, -7:, :], dim=-1)
            token_log_prob = log_prob[
                torch.arange(7), action_token_ids
            ].sum()

            losses_list.append(-(token_log_prob * adv))

            del logits
            torch.cuda.empty_cache()

    # 한 번에 평균 계산 (gradient graph 누적 방지)
    loss = torch.stack(losses_list).mean()
    return loss


# ─────────────────────────────────────────
# 단일 에피소드 실행 (평가용)
# ─────────────────────────────────────────

def run_episode(env, actor: ActorModel, instruction: str) -> dict:
    """
    단일 에피소드 실행 (성능 평가용).
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
    단일 태스크 GRPO 학습 (RLinf 방식 + LR 스케줄러 + 풍부한 보상함수).

    학습 루프 구조:
        에피소드 시작
        └── [1단계] collect_rollout() - no_grad
            → GROUP_SIZE개 trajectory 수집
            → reward_fn()으로 다중 보상 계산
        └── [2단계] compute_grpo_loss_from_trajectories() - gradient
            → rollout VRAM 해제 후 log_prob 재계산
            → GRPO loss → backward → optimizer.step()
            → scheduler.step()

    :param actor: ActorModel
    :param task_id: LIBERO-Long 태스크 인덱스
    :return: 학습 결과 통계
    """
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

        # ── [1단계] rollout 수집 (no_grad) ───────────
        trajectories = collect_rollout(actor, env, instruction, GROUP_SIZE)

        last_done = trajectories[0][-1]["done"]
        if last_done:
            successes.append(trajectories[0][-1]["reward"] >= 1.0)

        # ── [2단계] GRPO loss 계산 + 업데이트 ─────────
        optimizer.zero_grad()
        loss = compute_grpo_loss_from_trajectories(actor, trajectories)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

        losses.append(loss.item())
        current_lr = scheduler.get_last_lr()[0]

        # ── 매 에피소드 출력 확인 ────────────────────
        with torch.no_grad():
            sample_image         = get_image_from_obs(env.reset())
            critique, action_vec = actor.generate(
                image=sample_image,
                instruction=instruction
            )
            print(f"\n[에피소드 {episode+1} 출력]")
            print(f"  critique:      {critique if critique else '(없음)'}")
            print(f"  action_vector: {action_vec}")

        # ── 로그 출력 ─────────────────────────────────
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
    actor.qwen.gradient_checkpointing_enable()

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

    actor.qwen.save_pretrained(f"{SAVE_PATH}/final")
    print(f"최종 모델 저장: {SAVE_PATH}/final")

    actor.close()


if __name__ == "__main__":
    main()