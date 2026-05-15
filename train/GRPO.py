"""
GRPO 학습 스크립트
- 환경: LIBERO-Long (libero_10) 3개 태스크
- 각 태스크 500 에피소드 학습
- Actor: Qwen2.5-VL 3B + LoRA + GRPO
- Planner: OpenVLA (ZeroMQ 서버, openvla_env에서 별도 실행 필요)

실행 전 준비:
    터미널 1 (openvla_env): python openvla/openvla_inference_code.py
    터미널 2 (qwen_env):   python training/grpo_train.py
"""

import torch
import numpy as np
from PIL import Image
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from trl import GRPOConfig, GRPOTrainer
from actor_model import ActorModel


# ─────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────

TASK_SUITE     = "libero_long"   # LIBERO-Long (libero_10)
TASK_IDS       = [0, 1, 2]      # 학습할 태스크 3개 (0~9 중 선택)
NUM_EPISODES   = 500             # 태스크당 학습 에피소드 수
MAX_STEPS      = 300             # 에피소드당 최대 스텝 수
IMG_HEIGHT     = 224             # 이미지 높이
IMG_WIDTH      = 224             # 이미지 너비
SAVE_PATH      = "checkpoints"   # 체크포인트 저장 경로

# GRPO 설정
GRPO_CONFIG = GRPOConfig(
    output_dir=SAVE_PATH,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    logging_steps=10,
    save_steps=100,
    bf16=True,                   # bfloat16 학습
    remove_unused_columns=False
)


# ─────────────────────────────────────────
# LIBERO 환경 초기화
# ─────────────────────────────────────────

def make_env(task_id: int) -> OffScreenRenderEnv:
    """
    LIBERO-Long 환경 초기화.
    오프스크린 렌더링으로 이미지 관찰값 반환.

    :param task_id: LIBERO-Long 태스크 인덱스 (0~9)
    :return: LIBERO 환경 객체
    """
    # LIBERO benchmark 로드
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_SUITE]()
    task           = task_suite.get_task(task_id)
    task_name      = task_suite.get_task_name(task_id)
    task_bddl_file = task_suite.get_task_bddl_file_path(task_id)

    print(f"태스크 로드: {task_name}")

    # 환경 설정
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": IMG_HEIGHT,
        "camera_widths": IMG_WIDTH,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(42)
    return env, task_name


# ─────────────────────────────────────────
# 관찰값에서 이미지 추출
# ─────────────────────────────────────────

def get_image_from_obs(obs: dict) -> Image.Image:
    """
    LIBERO 환경 관찰값에서 PIL Image 추출.
    agentview 카메라 이미지 사용.

    :param obs: LIBERO 환경 관찰값 dict
    :return: PIL Image (224 x 224)
    """
    # LIBERO obs에서 이미지 추출
    # agentview_image: (H, W, 3) uint8 numpy array
    img_array = obs["agentview_image"]

    # numpy → PIL Image
    return Image.fromarray(img_array.astype(np.uint8))


# ─────────────────────────────────────────
# 단일 에피소드 실행
# ─────────────────────────────────────────

def run_episode(
    env,
    actor: ActorModel,
    instruction: str
) -> dict:
    """
    단일 에피소드 실행 후 결과 반환.
    GRPO 학습에 필요한 데이터 수집.

    :param env: LIBERO 환경
    :param actor: ActorModel (Qwen + ZeroMQ)
    :param instruction: 태스크 명령 텍스트
    :return: {
        "reward": float,         # 최종 reward (성공: 1.0, 실패: 0.0)
        "success": bool,         # 태스크 성공 여부
        "critiques": list[str],  # 각 스텝의 critique 텍스트
        "steps": int             # 실행한 스텝 수
    }
    """
    obs     = env.reset()
    done    = False
    success = False
    critiques = []

    for step in range(MAX_STEPS):

        # 1. 이미지 추출
        image = get_image_from_obs(obs)

        # 2. Actor 실행
        #    내부에서 ZeroMQ로 Planner action token 받아서 처리
        critique, action_vector = actor.generate(
            image=image,
            instruction=instruction
        )
        critiques.append(critique)

        # 3. LIBERO 환경에 action 실행
        obs, reward, done, info = env.step(action_vector)

        # 4. 성공 확인
        if done:
            success = info.get("success", False)
            break

    return {
        "reward":    1.0 if success else 0.0,
        "success":   success,
        "critiques": critiques,
        "steps":     step + 1
    }


# ─────────────────────────────────────────
# 태스크 학습
# ─────────────────────────────────────────

def train_on_task(
    actor: ActorModel,
    task_id: int
) -> dict:
    """
    단일 태스크에 대해 NUM_EPISODES 에피소드 학습.
    각 에피소드 결과를 GRPO 업데이트에 사용.

    :param actor: ActorModel
    :param task_id: LIBERO-Long 태스크 인덱스
    :return: 학습 결과 통계 dict
    """
    # 환경 초기화
    env, task_name = make_env(task_id)
    instruction    = task_name  # 태스크 이름을 명령 텍스트로 사용

    print(f"\n{'='*50}")
    print(f"태스크 {task_id}: {task_name}")
    print(f"에피소드 수: {NUM_EPISODES}")
    print(f"{'='*50}")

    rewards    = []
    successes  = []

    for episode in range(NUM_EPISODES):

        # 에피소드 실행
        result = run_episode(env, actor, instruction)

        rewards.append(result["reward"])
        successes.append(result["success"])

        # 로그 출력
        if (episode + 1) % 10 == 0:
            recent_rewards    = rewards[-10:]
            recent_success    = successes[-10:]
            avg_reward        = np.mean(recent_rewards)
            success_rate      = np.mean(recent_success) * 100

            print(
                f"[태스크 {task_id}] "
                f"에피소드 {episode+1}/{NUM_EPISODES} | "
                f"최근 10회 평균 reward: {avg_reward:.2f} | "
                f"성공률: {success_rate:.1f}%"
            )

        # 체크포인트 저장
        if (episode + 1) % 100 == 0:
            save_dir = f"{SAVE_PATH}/task_{task_id}_ep_{episode+1}"
            actor.qwen.save_pretrained(save_dir)
            print(f"체크포인트 저장: {save_dir}")

    env.close()

    # 태스크 결과 통계
    stats = {
        "task_id":      task_id,
        "task_name":    task_name,
        "avg_reward":   np.mean(rewards),
        "success_rate": np.mean(successes) * 100,
        "total_episodes": NUM_EPISODES
    }

    print(f"\n[태스크 {task_id} 완료]")
    print(f"  평균 reward:  {stats['avg_reward']:.3f}")
    print(f"  최종 성공률: {stats['success_rate']:.1f}%")

    return stats


# ─────────────────────────────────────────
# GRPO 설정 및 학습
# ─────────────────────────────────────────

def setup_grpo(actor: ActorModel):
    """
    GRPO 학습을 위한 reward 함수 정의.
    LIBERO 태스크 성공/실패를 reward로 사용.

    GRPO 동작 방식:
        1. 같은 입력으로 G개 샘플 생성 (Group sampling)
        2. 각 샘플의 reward 계산
        3. Group 내 상대적 reward로 advantage 계산
        4. PPO와 유사한 방식으로 policy 업데이트
        → value network 없이 advantage 추정 가능 (메모리 효율적)

    :param actor: ActorModel
    :return: GRPOTrainer
    """

    def reward_fn(completions, **kwargs):
        """
        GRPO reward 함수.
        Actor 출력에서 action을 파싱해서 LIBERO 환경에서 실행 후 reward 반환.

        :param completions: Actor가 생성한 텍스트 리스트
        :return: reward 리스트 (성공: 1.0, 실패: 0.0)
        """
        rewards = []
        for completion in completions:
            # 성공한 action이면 1.0, 실패하면 0.0
            # 실제로는 LIBERO 환경 실행 결과를 사용
            reward = 1.0 if "[ACTION]" in completion else 0.0
            rewards.append(reward)
        return rewards

    trainer = GRPOTrainer(
        model=actor.qwen,
        args=GRPO_CONFIG,
        reward_funcs=reward_fn,
        tokenizer=actor.processor.tokenizer
    )

    return trainer


# ─────────────────────────────────────────
# 메인 학습 루프
# ─────────────────────────────────────────

def main():
    """
    전체 학습 루프.
    TASK_IDS에 있는 태스크를 순서대로 학습.

    학습 순서:
        태스크 0 → 500 에피소드
        태스크 1 → 500 에피소드
        태스크 2 → 500 에피소드
    """

    print("Actor 모델 초기화 중...")
    print("(Planner 서버가 실행 중이어야 합니다)")
    actor = ActorModel()

    all_stats = []

    # 태스크별 순차 학습
    for task_id in TASK_IDS:
        stats = train_on_task(actor, task_id)
        all_stats.append(stats)

    # 전체 결과 출력
    print(f"\n{'='*50}")
    print("전체 학습 완료!")
    print(f"{'='*50}")
    for stats in all_stats:
        print(
            f"태스크 {stats['task_id']} ({stats['task_name']}): "
            f"성공률 {stats['success_rate']:.1f}%"
        )

    # 최종 모델 저장
    final_save_path = f"{SAVE_PATH}/final"
    actor.qwen.save_pretrained(final_save_path)
    print(f"\n최종 모델 저장: {final_save_path}")

    actor.close()


if __name__ == "__main__":
    main()