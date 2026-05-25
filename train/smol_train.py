"""
GRPO 학습 스크립트

[수정사항]
    forward() 시그니처 변경 대응:
        기존: forward(planner_action_tokens, cached_inputs, new_tokens)
        수정: forward(cached_inputs, new_tokens)
        이유: 액션이 프롬프트 텍스트에 포함되어 별도 전달 불필요

    GRPO loss 수정:
        기존: action token ID 범위로 구분 (49152~) → 불가 (정수 텍스트 토큰)
        수정: 위치 기반 가중치
              앞 70% = CRITIQUE 영역 → 0.3 가중치
              뒤 30% = ACTION 영역  → 0.7 가중치
              (ACTION: 7개 정수는 짧으므로 뒤쪽에 위치)

    파싱 실패 시 행동:
        기존: zeros → 항상 실패 → 보상 없음
        수정: 플래너 값 → 최소한 OpenVLA 수준 행동 유지

[OOM 수정] Vision Feature Caching 대응
    collect_rollout(): generate()의 8번째 반환값 image_hidden_states 언팩 및 저장
    compute_grpo_loss_from_trajectories():
        - rollout → 학습 전환 시 torch.cuda.synchronize() + empty_cache() 강제 수행
        - actor.forward()에 image_hidden_states 전달 → vision encoder 재실행 방지

실행:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train/smol_train.py
"""

import sys
import os

# [OOM 수정] 메모리 단편화 방지 설정
# expandable_segments: 조각난 free block 대신 OS에서 새 세그먼트 직접 할당
# max_split_size_mb: 큰 블록을 작은 조각으로 분리하지 않음
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256"
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

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

TASK_SUITE      = "libero_10"
TASK_IDS        = [0]
NUM_EPISODES    = 50
MAX_STEPS       = 30
IMG_HEIGHT      = 224
IMG_WIDTH       = 224
SAVE_PATH       = "checkpoints"

LEARNING_RATE   = 2e-4
LR_MIN          = 1e-6
WARMUP_EPISODES = 5

# 위치 기반 loss 가중치
# 생성 토큰의 앞 70%: critique 영역 (0.3)
# 생성 토큰의 뒤 30%: action 영역  (0.7)
CRITIQUE_RATIO  = 0.7   # 앞 70% 비율
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
            print("[optimizer] AdamW8bit (8GB 절약 모드)")
            return AdamW8bit(trainable, lr=LEARNING_RATE)
        except ImportError:
            print("[optimizer] bitsandbytes 미설치 → 표준 AdamW")
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

def reward_fn(action_vector: np.ndarray, info: dict) -> float:
    try:
        # [*수정*] 파싱 실패 상태가 감지되면 어떠한 경우라도 최하점(-1.0) 부여
        if info.get("parsing_failed", False):
            return -1.0

        reward = 0.0
        if info.get("success", False):
            reward += 1.0
        if np.any(np.isnan(action_vector)):
            return -1.0
        out_of_range = int(np.sum(np.abs(action_vector) > 1.0))
        if out_of_range > 0:
            reward -= 0.1 * out_of_range
        mag = float(np.linalg.norm(action_vector[:6]))
        if mag > 2.0:
            reward -= 0.2 * (mag - 2.0)
        return float(np.clip(reward, -1.0, 2.0))
    except Exception as e:
        print(f"[reward_fn 오류] {e}")
        return -1.0


# ─────────────────────────────────────────
# LR 스케줄러
# ─────────────────────────────────────────

def make_scheduler(optimizer, num_episodes: int):
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPISODES)
    cosine = CosineAnnealingLR(optimizer, T_max=num_episodes - WARMUP_EPISODES, eta_min=LR_MIN)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPISODES])


# ─────────────────────────────────────────
# [OOM 수정] VRAM 상태 출력 헬퍼
# ─────────────────────────────────────────

def log_vram(tag: str = ""):
    """allocated: 실제 사용 중 / reserved: PyTorch 캐시 보유 / peak: 최대 사용량"""
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
                # [*수정*] parsing_failed 상태값을 튜플 맨 뒤에서 추가로 받음
                # [OOM 수정] image_hidden_states를 8번째 값으로 추가 언팩
                (critique, action_vector, action_token_ids,
                 planner_tokens, cached_inputs,
                 new_tokens, parsing_failed, image_hidden_states) = actor.generate(
                    image=image, instruction=instruction
                )

            print(f"  [G{g+1} S{step+1}] {critique[:60] if critique else '(critique 없음)'}")
            print(f"  [G{g+1} S{step+1}] action: {np.round(action_vector, 3)}")
            print("=" * 50)

            obs, _, done, info = env.step(action_vector)

            # [*수정*] 보상 함수로 보내기 직전 info 변수에 파싱 결과를 안전하게 주입
            info["parsing_failed"] = parsing_failed
            reward = reward_fn(action_vector, info)

            group_traj.append({
                "cached_inputs":         cached_inputs,      # 프롬프트 (CPU)
                "new_tokens":            new_tokens,         # 생성 토큰 (CPU)
                "image_hidden_states":   image_hidden_states, # [OOM 수정] 캐싱된 image features (CPU)
                "planner_action_tokens": planner_tokens,     # 플래너 raw IDs
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
    위치 기반 critique/action 가중치.

    출력 토큰 구조:
        [CRITIQUE 텍스트 (긴 부분)] [ACTION: 숫자 7개 (짧은 부분)]

    ACTION 숫자는 뒤쪽 ~30% 토큰에 위치.
    앞 70% → critique weight, 뒤 30% → action weight.

    이렇게 하는 이유:
        action 결과(보상)가 학습의 핵심이므로 action 부분에 더 강한 gradient
        critique 텍스트도 함께 학습 → 해석 가능성 유지
    """
    # [OOM 수정] rollout에서 쌓인 단편화 residual 정리 후 학습 시작
    # rollout 20회의 generate()가 남긴 조각난 메모리 블록을 정리해서
    # backward() 시 280MB 연속 블록 확보를 보장
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # peak 추적 초기화 (에피소드별 peak 측정용)
    log_vram("학습 시작 전")              # [OOM 수정] 학습 시작 시점 VRAM 상태 확인

    rewards = torch.tensor(
        [traj[-1]["reward"] for traj in trajectories], dtype=torch.float32
    )

    if len(rewards) < 2 or rewards.std() < 1e-8:
        advantages = torch.zeros_like(rewards)
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss, total_steps = 0.0, 0

    for g, traj in enumerate(trajectories):
        adv = advantages[g].to("cuda")

        for step_data in traj:
            optimizer.zero_grad()

            new_tokens = step_data["new_tokens"].to("cuda")  # (N,)
            N = new_tokens.shape[0]

            # ── forward ────────────────────────────────────────────
            # [OOM 수정] image_hidden_states 전달 → vision encoder 재실행 방지
            logits, prompt_length = actor.forward(
                cached_inputs=step_data["cached_inputs"],
                new_tokens=new_tokens,
                image_hidden_states=step_data["image_hidden_states"]  # [OOM 수정] 추가
            )

            # ── logit 슬라이싱 ──────────────────────────────────────
            # logits[0, prompt_length-1] → new_tokens[0] 예측
            gen_logits = logits[0, prompt_length - 1 : -1, :]  # (N, vocab)
            log_prob   = torch.log_softmax(gen_logits, dim=-1)

            per_token_logp = log_prob[
                torch.arange(N, device="cuda"), new_tokens
            ]  # (N,)

            # ── 위치 기반 가중치 ─────────────────────────────────────
            # 앞 critique_boundary개: CRITIQUE 영역 (낮은 가중치)
            # 뒤 나머지:              ACTION 영역 (높은 가중치)
            critique_boundary = max(1, int(N * CRITIQUE_RATIO))
            action_boundary   = N - critique_boundary

            weights = torch.ones(N, device="cuda")
            weights[:critique_boundary]  = CRITIQUE_WEIGHT
            weights[critique_boundary:]  = ACTION_WEIGHT

            # 정규화 (총 토큰 수로 나눔)
            weighted_logp = (per_token_logp * weights).sum() / N

            # ── GRPO loss ───────────────────────────────────────────
            step_loss = -(weighted_logp * adv)

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss  += step_loss.item()
            total_steps += 1

            if total_steps % 10 == 0:
                print(
                    f"  [loss] total={step_loss.item():.4f} | "
                    f"adv={adv.item():.3f} | "
                    f"critique_len={critique_boundary} action_len={action_boundary}"
                )

            del logits, gen_logits, log_prob, per_token_logp, weights, step_loss, new_tokens
            torch.cuda.empty_cache()

    log_vram("학습 종료 후")  # [OOM 수정] 학습 후 VRAM 상태 확인 (누수 감지용)
    return total_loss / max(total_steps, 1)


# ─────────────────────────────────────────
# 평가
# ─────────────────────────────────────────

def run_episode(env, actor: ActorModel, instruction: str) -> dict:
    obs, success = env.reset(), False
    for step in range(MAX_STEPS):
        with torch.no_grad():
            # [참고] Python의 *_ 언패킹 문법 덕분에 추가된 parsing_failed 변수 처리에 코드 수정이 필요 없습니다.
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

        if (episode + 1) % 100 == 0:
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

    print(f"\n[태스크 {task_id} 완료]")
    print(f"  평균 loss:   {stats['avg_loss']:.4f}")
    print(f"  최종 성공률: {stats['success_rate']:.1f}%")

    return stats


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────

def main():
    # [OOM 수정] CUDA 설정값 확인 출력
    print(f"[CUDA 설정] PYTORCH_CUDA_ALLOC_CONF = "
          f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '미설정')}")
    print("Actor 모델 초기화 중...")
    print("Planner 서버(openvla_inference_code.py)가 실행 중이어야 합니다.")

    actor = ActorModel()
    actor.smol.gradient_checkpointing_enable()

    log_vram("모델 로드 후")  # [OOM 수정] 초기 VRAM 기준점 출력

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