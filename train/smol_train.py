"""
GRPO 학습 스크립트 (RLinf 방식 + LR 스케줄러 + 풍부한 보상함수)
- 환경: LIBERO-Long 3개 태스크, 각 300 에피소드
- rollout 단계와 학습 단계 완전 분리 (VRAM 효율화 핵심)
- Linear Warmup + Cosine Annealing LR 스케줄러
- 다중 보상 신호 적용

OOM 수정 핵심:
    기존: 모든 스텝 loss를 모아서 한 번에 backward
          → gradient graph가 스텝 수만큼 누적 → VRAM 폭발
    수정: 스텝마다 즉시 backward + optimizer.step()
          → gradient graph를 스텝마다 해제 → VRAM 절약

실행 전 준비:
    터미널 1 (openvla_env): python openvla_planner/openvla_inference_code.py
    터미널 2 (qwen_env):
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        python train/smol_train.py
"""

import sys
import os

# SmolVLM_actor 폴더를 Python path에 추가
# train/ 폴더에서 실행할 때 상위 폴더의 SmolVLM_actor를 찾기 위함
sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

import torch
import numpy as np
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# SmolVLM_actor 폴더의 smol_actor_model에서 ActorModel 임포트
from SmolVLM_actor.smol_actor_model import ActorModel


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

TASK_SUITE      = "libero_10"   # LIBERO benchmark 키 이름
TASK_IDS        = [0, 1, 2]    # 학습할 태스크 인덱스 (0~9 중 3개)
NUM_EPISODES    = 300           # 태스크당 학습 에피소드 수
MAX_STEPS       = 20            # 에피소드당 최대 스텝 수 (VRAM 절약)
IMG_HEIGHT      = 224           # LIBERO 렌더링 이미지 높이
IMG_WIDTH       = 224           # LIBERO 렌더링 이미지 너비
SAVE_PATH       = "checkpoints" # 체크포인트 저장 경로
GROUP_SIZE      = 5             # GRPO group sampling 수 (VRAM 절약으로 1로 설정)

# LR 스케줄러 설정
LEARNING_RATE   = 1e-4          # 최대 학습률 (warmup 후 도달)
LR_MIN          = 1e-6          # Cosine Annealing 최소 학습률
WARMUP_EPISODES = 20            # warmup 에피소드 수


# ─────────────────────────────────────────
# LIBERO 환경 초기화
# ─────────────────────────────────────────

def make_env(task_id: int):
    """
    LIBERO-Long 환경 초기화.
    OffScreenRenderEnv를 사용해서 화면 없이 이미지 렌더링.

    :param task_id: 태스크 인덱스 (0~9)
    :return: (env, task_name)
    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_SUITE]()
    task_names     = task_suite.get_task_names()    # 전체 태스크 이름 리스트
    task_name      = task_names[task_id]            # 인덱스로 태스크 이름 선택
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
    """
    LIBERO 환경 관찰값에서 PIL Image 추출.
    agentview 카메라 이미지를 사용.

    :param obs: LIBERO 환경 관찰값 dict
    :return: PIL Image (224 x 224)
    """
    return Image.fromarray(obs["agentview_image"].astype(np.uint8))


# ─────────────────────────────────────────
# 보상 함수
# ─────────────────────────────────────────

def reward_fn(action_vector: np.ndarray, obs, info: dict, done: bool) -> float:
    """
    다중 보상 신호를 결합한 보상 함수.
    여기만 수정하면 보상 설계 변경 가능.

    보상 구성:
        +1.0               태스크 성공
        -0.5               충돌 발생
        -0.01 × overtime   100 스텝 초과 시 시간 패널티
        +0.3 × stability   end-effector 속도 안정성 보상
        -1.0               action_vector에 NaN 포함 시 패널티
        clip [-1.0, 2.0]   최종 보상 클리핑

    :param action_vector: 실행된 action vector (7,)
    :param obs: 환경 관찰값
    :param info: 환경 info dict
    :param done: 에피소드 종료 여부
    :return: 보상값 (float)
    """
    total_reward = 0.0

    try:
        # 1. 태스크 성공 보상 (+1.0)
        success = info.get("success", False)
        if success:
            total_reward += 1.0

        # 2. 충돌 패널티 (-0.5)
        collision = info.get("collision", False)
        if collision:
            total_reward -= 0.5

        # 3. 시간 패널티 (100 스텝 이후부터 -0.01/스텝)
        current_step = info.get("step", 0)
        free_steps   = 100
        if current_step > free_steps:
            overtime      = current_step - free_steps
            total_reward -= 0.01 * overtime

        # 4. end-effector 안정성 보상 (+0.3 × stability)
        ee_velocity      = info.get("ee_velocity", 0.0)
        stability_reward = max(0.0, 1.0 - abs(ee_velocity))
        total_reward    += 0.3 * stability_reward

        # 5. NaN 패널티 (-1.0)
        if np.any(np.isnan(action_vector)):
            total_reward -= 1.0

        # 6. 최종 클리핑 [-1.0, 2.0]
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

    GRPO는 초반에 reward가 sparse해서 학습이 불안정할 수 있음.
    warmup으로 초반 lr을 낮게 시작해서 안정적으로 학습.

    학습 단계:
        1. Warmup (WARMUP_EPISODES 동안)
           lr: LEARNING_RATE × 0.1 → LEARNING_RATE 선형 증가

        2. Cosine Annealing (나머지 에피소드 동안)
           lr: LEARNING_RATE → LR_MIN 코사인 감소

    :param optimizer: AdamW optimizer
    :param num_episodes: 전체 에피소드 수
    :return: SequentialLR 스케줄러
    """
    # warmup: lr을 0.1배에서 시작해서 선형으로 증가
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_EPISODES
    )

    # cosine: lr을 코사인 곡선으로 감소
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_episodes - WARMUP_EPISODES,
        eta_min=LR_MIN
    )

    # warmup → cosine 순서로 실행
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

    핵심: torch.no_grad()로 inference만 수행
        → activation 저장 안 함 → VRAM 절약
        → 수집된 trajectory는 학습 단계에서 재사용

    같은 초기 상태에서 group_size개 trajectory를 수집해서
    GRPO의 group sampling 구현.

    :param actor: ActorModel (SmolVLM2 기반)
    :param env: LIBERO 환경
    :param instruction: 태스크 명령 텍스트
    :param group_size: 수집할 trajectory 수
    :return: group_size개 trajectory list
    """
    trajectories = []

    obs = env.reset()

    # 초기 상태 저장 (group_size > 1일 때 같은 상태에서 여러 번 샘플링)
    try:
        init_state        = env.get_state()
        use_state_restore = True
    except AttributeError:
        use_state_restore = False

    for g in range(group_size):

        # 첫 번째 이후 group에서는 초기 상태로 복원
        if g > 0 and use_state_restore:
            env.set_state(init_state)

        group_traj = []

        for step in range(MAX_STEPS):
            image = get_image_from_obs(obs)

            # ── inference only (no_grad) ──────────────────
            # gradient 없이 inference만 수행 → VRAM 절약
            with torch.no_grad():
                # SmolVLM2 forward → logits (batch, seq_len, vocab_size)
                logits = actor.forward(image, instruction)

                # 마지막 7개 logits에서 action token 샘플링
                action_token_ids = torch.multinomial(
                    torch.softmax(logits[0, -7:, :], dim=-1),
                    num_samples=1
                ).squeeze(-1)  # shape: (7,)

                # SmolVLM2 vocab 기준 token ID → 연속값 action vector 변환
                action_vector = actor.action_tokenizer.decode_token_ids_to_actions(
                    action_token_ids.cpu().numpy()
                )

                # logits 즉시 삭제로 VRAM 해제
                del logits
                torch.cuda.empty_cache()

            # LIBERO 환경에 action 실행
            obs, _, done, info = env.step(action_vector)

            # 다중 보상 함수로 reward 계산
            reward = reward_fn(action_vector, obs, info, done)

            # trajectory에 스텝 데이터 저장
            # action_token_ids는 학습 단계에서 log_prob 재계산에 사용
            group_traj.append({
                "image":            image,
                "instruction":      instruction,
                "action_token_ids": action_token_ids.cpu(),  # CPU로 이동 (메모리 절약)
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
    trajectories: list,
    optimizer: AdamW
) -> float:
    """
    수집된 trajectory로 GRPO loss 계산 + 스텝마다 즉시 업데이트.

    OOM 수정 핵심:
        기존: losses_list에 모아서 stack().mean().backward()
              → 모든 스텝의 gradient graph가 동시에 메모리에 유지 → OOM
        수정: 스텝마다 즉시 backward() + optimizer.step()
              → 각 스텝의 gradient graph가 즉시 해제 → VRAM 절약

    GRPO 동작:
        1. 각 trajectory의 마지막 reward 수집
        2. group 내 상대적 advantage 계산
           advantage = (reward - mean) / std
           GROUP_SIZE=1이면 advantage=0 → 업데이트 없음
        3. 각 스텝에서 rollout 때 선택한 action의 log_prob 재계산
        4. step_loss = -(log_prob × advantage) → 즉시 backward

    :param actor: ActorModel
    :param trajectories: collect_rollout() 반환값
    :param optimizer: AdamW optimizer (스텝마다 즉시 업데이트)
    :return: 평균 loss (float, logging용)
    """
    # 각 trajectory의 마지막 reward 수집
    rewards = torch.tensor(
        [traj[-1]["reward"] for traj in trajectories],
        dtype=torch.float32
    )

    # advantage 계산 (group 내 상대적 reward)
    # GROUP_SIZE=1이면 advantage=0이 되어 실질적 업데이트 없음
    # → GROUP_SIZE를 2 이상으로 늘리면 학습 효과적 (VRAM 해결 후 권장)
    if len(rewards) < 2 or rewards.std() < 1e-8:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss  = 0.0   # logging용 누적 loss (gradient 없는 scalar)
    total_steps = 0

    for g, traj in enumerate(trajectories):
        adv = advantages[g].to("cuda")

        for step_data in traj:

            # ── 스텝마다 즉시 backward (OOM 수정 핵심) ──
            # gradient 초기화: 이전 스텝의 gradient가 남지 않도록
            optimizer.zero_grad()

            # rollout에서 저장한 image, instruction으로 log_prob 재계산
            # 이번엔 gradient 있음 (no_grad 없음)
            logits = actor.forward(
                step_data["image"],
                step_data["instruction"]
            )

            # rollout에서 선택한 action token ID로 log_prob 계산
            action_token_ids = step_data["action_token_ids"].to("cuda")
            log_prob = torch.log_softmax(logits[0, -7:, :], dim=-1)
            token_log_prob = log_prob[
                torch.arange(7), action_token_ids
            ].sum()  # 7개 action token의 log_prob 합산

            # GRPO loss: -(log_prob × advantage)
            # advantage > 0: 이 action을 더 선택하도록 학습 (log_prob 증가)
            # advantage < 0: 이 action을 덜 선택하도록 학습 (log_prob 감소)
            step_loss = -(token_log_prob * adv)

            # 즉시 backward → 이 스텝의 gradient graph만 메모리에 올라감
            # backward 후 gradient graph 자동 해제 → 다음 스텝에 영향 없음
            step_loss.backward()

            # gradient clipping: 너무 큰 gradient로 인한 학습 불안정 방지
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

            # 이 스텝의 gradient로 파라미터 즉시 업데이트
            optimizer.step()

            # logging용 loss 누적 (gradient 없는 scalar)
            total_loss  += step_loss.item()
            total_steps += 1

            # logits, step_loss 즉시 삭제 → VRAM 해제
            del logits, step_loss
            torch.cuda.empty_cache()

    # 평균 loss 반환 (logging용, 이미 모든 backward/step 완료)
    return total_loss / max(total_steps, 1)


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
    단일 태스크 GRPO 학습 (RLinf 방식 + LR 스케줄러 + 풍부한 보상함수).

    학습 루프 구조:
        에피소드 시작
        └── [1단계] collect_rollout() - no_grad
            → GROUP_SIZE개 trajectory 수집
            → reward_fn()으로 다중 보상 계산
            → VRAM: activation 저장 안 함 → 낮은 VRAM 사용

        └── [2단계] compute_grpo_loss_from_trajectories() - gradient
            → 스텝마다 즉시 backward + optimizer.step() (OOM 수정)
            → 각 스텝의 gradient graph가 즉시 해제
            → backward/step은 함수 내부에서 처리

        └── scheduler.step() → lr 업데이트 (에피소드 단위)

        └── 매 에피소드 generate()로 출력 확인

    :param actor: ActorModel (SmolVLM2 기반)
    :param task_id: LIBERO-Long 태스크 인덱스
    :return: 학습 결과 통계 dict
    """
    env, task_name = make_env(task_id)
    instruction    = task_name

    print(f"\n{'='*50}")
    print(f"태스크 {task_id}: {task_name}")
    print(f"에피소드: {NUM_EPISODES} | Group size: {GROUP_SIZE} | Max steps: {MAX_STEPS}")
    print(f"LR: {LEARNING_RATE} (warmup {WARMUP_EPISODES}ep) → {LR_MIN} (cosine)")
    print(f"{'='*50}")

    # LoRA + Projection Layer 파라미터만 학습
    # requires_grad=True인 파라미터만 optimizer에 전달
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, actor.parameters()),
        lr=LEARNING_RATE
    )

    # Linear Warmup + Cosine Annealing 스케줄러
    scheduler = make_scheduler(optimizer, NUM_EPISODES)

    losses    = []
    successes = []

    for episode in range(NUM_EPISODES):

        # ── [1단계] rollout 수집 (no_grad) ───────────
        # inference만 수행 → VRAM 낮음
        trajectories = collect_rollout(actor, env, instruction, GROUP_SIZE)

        # 에피소드 성공 여부 확인
        last_done = trajectories[0][-1]["done"]
        if last_done:
            successes.append(trajectories[0][-1]["reward"] >= 1.0)

        # ── [2단계] GRPO loss 계산 + 즉시 업데이트 ────
        # optimizer를 함수에 넘겨서 스텝마다 즉시 업데이트
        # backward/zero_grad/step은 함수 내부에서 처리
        # → 여기서 별도로 호출하면 안 됨 (중복 업데이트 방지)
        loss_val = compute_grpo_loss_from_trajectories(
            actor, trajectories, optimizer
        )

        # LR 스케줄러 업데이트 (에피소드마다 한 번)
        # optimizer.step()은 함수 내부에서 스텝마다 호출했지만
        # scheduler.step()은 에피소드 단위로 한 번만 호출
        scheduler.step()
        torch.cuda.empty_cache()

        losses.append(loss_val)
        current_lr = scheduler.get_last_lr()[0]

        # ── 매 에피소드 출력 확인 ────────────────────
        # generate()로 현재 Actor가 무엇을 출력하는지 확인
        with torch.no_grad():
            sample_image         = get_image_from_obs(env.reset())
            critique, action_vec = actor.generate(
                image=sample_image,
                instruction=instruction
            )
            print(f"\n[에피소드 {episode+1} 출력]")
            print(f"  critique:      {critique if critique else '(없음)'}")
            print(f"  action_vector: {action_vec}")

        # ── 로그 출력 (10 에피소드마다) ───────────────
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

        # ── 체크포인트 저장 (100 에피소드마다) ──────────
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
    """
    전체 학습 루프.
    태스크 0 → 1 → 2 순서로 300 에피소드씩 학습.
    """
    print("Actor 모델 초기화 중...")
    print("Planner 서버(openvla_inference_code.py)가 실행 중이어야 합니다.")

    actor = ActorModel()

    # gradient checkpointing: activation을 저장하지 않고
    # backward 시 재계산해서 VRAM 절약 (속도는 약간 느려짐)
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

    # 최종 모델 저장
    actor.smol.save_pretrained(f"{SAVE_PATH}/final")
    print(f"최종 모델 저장: {SAVE_PATH}/final")

    actor.close()


if __name__ == "__main__":
    main()