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
        python train/train.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")
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

TASK_SUITE      = "libero_10"
TASK_IDS        = [0, 1, 2]
NUM_EPISODES    = 100
MAX_STEPS       = 20
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
SAVE_PATH       = "checkpoints"
GROUP_SIZE      = 5

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
        +1.0               태스크 성공
        -0.5               충돌 발생
        -0.01 × overtime   100 스텝 초과 시 시간 패널티
        +0.3 × stability   end-effector 속도 안정성 보상
        -1.0               action_vector에 NaN 포함 시 패널티
        clip [-1.0, 2.0]   최종 보상 클리핑
    """
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
    """
    Linear Warmup + Cosine Annealing 스케줄러 생성.
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

    핵심 변경:
        기존: forward() → logits에서 바로 action token 샘플링
              → critique 텍스트 없음 → 비판적 평가 없음
        수정: generate() 사용
              → [CRITIQUE] 텍스트 생성 → [ACTION] action token 생성
              → 원래 의도한 비판적 재평가 구조

    generate()가 반환하는 action_token_ids를
    학습 단계에서 log_prob 재계산에 사용.

    :param actor: ActorModel (Qwen 기반)
    :param env: LIBERO 환경
    :param instruction: 태스크 명령 텍스트
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

            # generate()로 critique + action token + action vector 생성
            # no_grad: rollout 단계는 inference만 수행 → VRAM 절약
            with torch.no_grad():
                # generate()가 4개 반환:
                # critique:              Actor의 비판 텍스트
                # action_vector:         환경에 실행할 연속값 (7,)
                # action_token_ids:      학습 단계 log_prob 재계산용 (7,)
                # planner_action_tokens: 학습 단계 forward() 재사용용 → OpenVLA 중복 호출 방지
                critique, action_vector, action_token_ids, planner_tokens = actor.generate(
                    image=image,
                    instruction=instruction
                )

            # OOM 방지: rollout 단계는 VRAM 절약이 중요 → 매 스텝마다 불필요한 텐서 해제 + 캐시 정리    
            torch.cuda.empty_cache()

            # 매 스텝 critique + action 출력
            print(f"  [스텝 {step+1}] critique:      {critique[:60] if critique else '(없음)'}")
            print(f"  [스텝 {step+1}] action_vector: {action_vector}")
            print(f"  [스텝 {step+1}] action_tokens: {action_token_ids}")

            # LIBERO 환경에 action 실행
            obs, _, done, info = env.step(action_vector)
            reward = reward_fn(action_vector, obs, info, done)

            # trajectory 저장
            # action_token_ids: 학습 단계에서 log_prob 재계산에 사용
            group_traj.append({
                "image":                 image,
                "instruction":           instruction,
                "action_token_ids":      torch.tensor(action_token_ids, dtype=torch.long),
                "planner_action_tokens": planner_tokens,   # OpenVLA 재호출 방지용
                "cached_inputs":         cached_inputs,    # processor 재호출 방지용
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
    """
    수집된 trajectory로 GRPO loss 계산 + 스텝마다 즉시 업데이트.

    OOM 수정:
        스텝마다 즉시 backward() + optimizer.step()
        → 각 스텝 gradient graph 즉시 해제 → VRAM 절약

    rollout에서 generate()가 선택한 action_token_ids로
    log_prob을 재계산해서 GRPO loss 계산.

    :param actor: ActorModel
    :param trajectories: collect_rollout() 반환값
    :param optimizer: AdamW (스텝마다 즉시 업데이트)
    :return: 평균 loss (float, logging용)
    """
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

            # 스텝마다 즉시 backward (OOM 수정 핵심)
            optimizer.zero_grad()

            # rollout에서 저장한 planner_action_tokens 재사용
            # → OpenVLA 중복 호출 없이 동일한 Planner token 사용
            logits, input_ids = actor.forward(
                step_data["image"],
                step_data["instruction"],
                planner_action_tokens=step_data["planner_action_tokens"],
                cached_inputs=step_data["cached_inputs"]  # processor 재호출 없음
            )

            # [ACTION] 토큰 위치를 정확히 찾아서 log_prob 계산
            # logits[-7:]이 아니라 [ACTION] 태그 다음 7개 위치 사용
            action_token_ids = step_data["action_token_ids"].to("cuda")
            try:
                action_tag_id  = actor.processor.tokenizer.convert_tokens_to_ids("[ACTION]")
                input_ids_list = input_ids[0].tolist()
                action_tag_pos = input_ids_list.index(action_tag_id)
                # [ACTION] 태그 다음 위치부터 7개 logits 사용
                log_prob = torch.log_softmax(
                    logits[0, action_tag_pos:action_tag_pos+7, :], dim=-1
                )
            except (ValueError, Exception):
                # [ACTION] 태그를 못 찾으면 기존 방식으로 fallback
                log_prob = torch.log_softmax(logits[0, -7:, :], dim=-1)

            token_log_prob = log_prob[
                torch.arange(7), action_token_ids
            ].sum()

            step_loss = -(token_log_prob * adv)

            # 즉시 backward → gradient graph 해제
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += step_loss.item()
            total_steps += 1

            del logits, input_ids, step_loss
            torch.cuda.empty_cache()

    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────
# 단일 에피소드 실행 (평가용)
# ─────────────────────────────────────────

def run_episode(env, actor: ActorModel, instruction: str) -> dict:
    """
    단일 에피소드 실행 (성능 평가용).
    gradient 계산 없음.
    """
    obs     = env.reset()
    success = False

    for step in range(MAX_STEPS):
        image = get_image_from_obs(obs)

        with torch.no_grad():
            _, action_vector, _, _ = actor.generate(
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

    학습 루프 구조:
        [1단계] collect_rollout() - no_grad
            → generate()로 critique + action token 생성
            → 매 스텝 critique/action 출력
            → reward_fn()으로 보상 계산

        [2단계] compute_grpo_loss_from_trajectories() - gradient
            → 스텝마다 즉시 backward + optimizer.step()
            → scheduler.step() (에피소드 단위)
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

        print(f"\n[에피소드 {episode+1}] rollout 수집 중...")

        # [1단계] rollout 수집 (no_grad)
        # generate()로 critique + action token 생성 + 매 스텝 출력
        trajectories = collect_rollout(actor, env, instruction, GROUP_SIZE)

        last_done = trajectories[0][-1]["done"]
        if last_done:
            successes.append(trajectories[0][-1]["reward"] >= 1.0)

        # [2단계] GRPO loss 계산 + 즉시 업데이트
        loss_val = compute_grpo_loss_from_trajectories(
            actor, trajectories, optimizer
        )

        # LR 스케줄러 (에피소드 단위)
        scheduler.step()
        torch.cuda.empty_cache()

        losses.append(loss_val)
        current_lr = scheduler.get_last_lr()[0]

        # 로그 출력 (10 에피소드마다)
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

        # 체크포인트 저장 (100 에피소드마다)
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