"""
smol_sft_stage3.py — Stage 3: 성공/실패 예측 + 액션 생성 학습

[목적]
    OpenVLA rollout → 성공/실패 레이블 수집 (53.7% 성공률 → 균형잡힌 데이터)

    SUCCESS 에피소드: scene_text + SUCCESS + [ACTION] planner_action [/ACTION]
                     손실: SUCCESS 토큰 + 액션 7개 (텍스트 free)
                     → 액션 토큰 생성 능력 + 성공 인식 동시 학습

    FAILURE 에피소드: scene_text + FAILURE
                     손실: FAILURE 토큰만 (텍스트 free)
                     → 실패 인식 학습 (틀린 액션은 강화하지 않음)

[GRPO 연결]
    모델이 "이 장면 + 이 액션 = SUCCESS/FAILURE" 판단 가능
    → GRPO에서 "FAILURE니까 액션 수정" 자연스럽게 학습

로드: checkpoints/sft_stage2
저장: checkpoints/sft_stage3
"""

import os
import sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from safetensors.torch import load_file

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from SmolVLM_actor.smol_actor_model import ActorModel

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
TASK_SUITE           = "libero_spatial"
TASK_IDS             = [0, 1, 2, 3, 4]
EPISODES_PER_TASK    = 4          # 8 → 4
MAX_STEPS_PER_EP     = 150        # 250 → 150
TRAIN_EPOCHS         = 3          # 수집 데이터 반복 학습
LEARNING_RATE        = 1e-5
IMG_HEIGHT           = 224
IMG_WIDTH            = 224
LOAD_PATH            = "checkpoints/sft_stage2_5"
SAVE_PATH            = "checkpoints/sft_stage3"
FALLBACK_TEXT        = "I observe the scene and evaluate the proposed action."


# ─────────────────────────────────────────
# 체크포인트 로드
# ─────────────────────────────────────────

def load_checkpoint(actor: ActorModel, path: str):
    print(f"[체크포인트 로드] {path}")
    sf_files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    if not sf_files:
        print("  경고: safetensors 없음, 기본 가중치 사용")
        return
    state_dict = {}
    for f in sf_files:
        state_dict.update(load_file(os.path.join(path, f)))
    missing, unexpected = actor.smol.load_state_dict(state_dict, strict=False)
    print(f"  완료 | missing={len(missing)} unexpected={len(unexpected)}")


# ─────────────────────────────────────────
# LIBERO 환경
# ─────────────────────────────────────────

def make_env(task_id: int):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite     = benchmark_dict[TASK_SUITE]()
    task_name      = task_suite.get_task_names()[task_id]
    task_bddl_file = task_suite.get_task_bddl_file_path(task_id)
    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_bddl_file,
        "camera_heights": IMG_HEIGHT,
        "camera_widths":  IMG_WIDTH,
    })
    env.seed(42 + task_id)
    return env, task_name


def get_image(obs):
    return Image.fromarray(obs["agentview_image"].astype(np.uint8))


# ─────────────────────────────────────────
# Scene text 사전 수집
# ─────────────────────────────────────────

def get_scene_text(actor: ActorModel, image: Image.Image,
                   instruction: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Briefly describe the scene for: {instruction}"}
        ]
    }]
    inputs = actor.processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt"
    )
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    try:
        with torch.no_grad():
            with actor.smol.disable_adapter():
                out = actor.smol.generate(
                    **inputs, max_new_tokens=40,
                    do_sample=True, temperature=0.7,
                )
        prompt_len = inputs["input_ids"].shape[1]
        text = actor.processor.decode(
            out[0][prompt_len:], skip_special_tokens=True
        ).strip()
        del out
    except Exception:
        text = ""
    del inputs
    torch.cuda.empty_cache()
    return text if text else FALLBACK_TEXT


def precompute_scene_texts(actor, env, instruction, n=60):
    print(f"  [scene_text 사전 수집] {n}개...")
    texts = []
    obs = env.reset()
    for _ in range(n):
        image = get_image(obs)
        texts.append(get_scene_text(actor, image, instruction))
        obs, _, done, _ = env.step(np.zeros(7))
        if done:
            obs = env.reset()
    torch.cuda.empty_cache()
    print(f"  [scene_text 사전 수집] 완료 | 고유: {len(set(texts))}/{n}")
    return texts


# ─────────────────────────────────────────
# IHS 캡처 (ZeroMQ 없이, 저장된 planner_tokens 사용)
# ─────────────────────────────────────────

def capture_ihs(actor: ActorModel, image: Image.Image,
                instruction: str, planner_tokens: np.ndarray):
    """
    저장된 planner_tokens으로 IHS 캡처.
    max_new_tokens=1로 빠르게 vision encoder만 실행.
    ZeroMQ 플래너 재호출 없음.
    """
    planner_bin   = actor.action_tokenizer.openvla_ids_to_bin_indices(
        np.array(planner_tokens)
    )
    cached_inputs = actor._apply_chat_template(image, instruction, planner_bin)

    input_ids      = cached_inputs["input_ids"].to("cuda")
    attention_mask = cached_inputs["attention_mask"].to("cuda")
    pixel_values   = cached_inputs["pixel_values"].to("cuda", dtype=torch.bfloat16).detach()
    pixel_values.requires_grad_(False)

    action_embeds      = actor.action_tokenizer.embed_action_tokens(
        torch.tensor(planner_tokens, dtype=torch.long)
    )
    critique_token_len = cached_inputs.get("critique_token_len", 0)
    insert_pos         = input_ids.shape[1] - critique_token_len

    try:
        actor._action_embeds_buffer[0] = action_embeds
        actor._action_insert_pos[0]    = insert_pos
        actor._capture_ihs             = True
        actor._ihs_cache[0]            = None
        with torch.no_grad():
            actor.smol.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=actor.processor.tokenizer.eos_token_id
            )
    finally:
        actor._action_embeds_buffer[0] = None
        actor._action_insert_pos[0]    = None
        actor._capture_ihs             = False

    ihs = actor._ihs_cache[0]
    del input_ids, attention_mask, pixel_values
    torch.cuda.empty_cache()
    return cached_inputs, ihs


# ─────────────────────────────────────────
# Rollout 데이터 수집
# ─────────────────────────────────────────

def collect_rollout(actor: ActorModel, env, instruction: str,
                    n_episodes: int) -> list:
    """
    OpenVLA rollout 실행 → (image_np, planner_tokens, success) 수집.
    저장 크기 최소화: numpy 이미지 + 7개 정수 + bool.
    """
    dataset = []
    success_count = 0

    for ep in range(n_episodes):
        obs       = env.reset()
        ep_steps  = []
        success   = False

        for step in range(MAX_STEPS_PER_EP):
            image         = get_image(obs)
            planner_tokens = actor.get_planner_action_tokens(image, instruction)
            ep_steps.append((np.array(image), planner_tokens.copy()))

            planner_bin = actor.action_tokenizer.openvla_ids_to_bin_indices(planner_tokens)
            action      = actor.action_tokenizer.bin_indices_to_continuous(planner_bin)
            obs, _, done, info = env.step(action)

            if step % 50 == 0:
                print(f"    step {step:3d}/{MAX_STEPS_PER_EP} ...", flush=True)

            if done:
                success = bool(info.get("success", False))
                break

        for img_np, tokens in ep_steps:
            dataset.append((img_np, tokens, success))

        success_count += int(success)
        print(f"  ep {ep+1:2d}/{n_episodes} | steps={len(ep_steps):3d} | "
              f"{'✓ SUCCESS' if success else '✗ FAILURE'}")

    print(f"  수집 완료 | SUCCESS={success_count} FAILURE={n_episodes-success_count} "
          f"| 총 {len(dataset)} 스텝")
    return dataset


# ─────────────────────────────────────────
# SFT 스텝
# ─────────────────────────────────────────

def sft_step(actor, optimizer, cached_inputs,
             planner_tokens, target_ids, loss_mask, ihs):
    optimizer.zero_grad()

    if loss_mask.sum() < 1:
        return 0.0

    target_gpu    = target_ids.to("cuda")
    loss_mask_gpu = loss_mask.to("cuda")

    logits, prompt_length = actor.forward(
        cached_inputs=cached_inputs,
        new_tokens=target_gpu[0],
        planner_action_tokens=planner_tokens,
        image_hidden_states=ihs
    )

    gen_logits     = logits[0, prompt_length - 1 : -1]
    per_token_loss = F.cross_entropy(gen_logits, target_gpu[0], reduction="none")
    loss           = (per_token_loss * loss_mask_gpu).sum() / loss_mask_gpu.sum()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    optimizer.step()

    val = loss.item()
    del logits, gen_logits, per_token_loss, loss, target_gpu, loss_mask_gpu
    torch.cuda.empty_cache()
    return val


# ─────────────────────────────────────────
# 타겟 + 손실 마스크 생성
# ─────────────────────────────────────────

def make_target(actor, planner_tokens, scene_text, success):
    """
    SUCCESS: scene_text + SUCCESS + [ACTION] 7tokens [/ACTION] + EOS
             손실: SUCCESS 토큰부터 EOS까지 (텍스트 free)

    FAILURE: scene_text + FAILURE + EOS
             손실: FAILURE 토큰 + EOS만 (텍스트 free)
    """
    smol_start = len(actor.processor.tokenizer) - 256
    tokenizer  = actor.processor.tokenizer
    eos        = tokenizer.eos_token

    if success:
        planner_bin = actor.action_tokenizer.openvla_ids_to_bin_indices(
            np.array(planner_tokens)
        )
        action_str  = " ".join([f"<action_{b}>" for b in planner_bin])
        # OpenVLA 방식: SUCCESS 다음 바로 액션 토큰 7개 + EOS
        target_text = f"{scene_text}\nSUCCESS\n{action_str}{eos}"
    else:
        target_text = f"{scene_text}\nFAILURE{eos}"

    target_ids = tokenizer(
        target_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"]  # (1, T)

    target_np = target_ids.numpy()[0]

    # SUCCESS/FAILURE 토큰 위치 찾기
    success_ids = tokenizer("SUCCESS", add_special_tokens=False)["input_ids"]
    failure_ids = tokenizer("FAILURE", add_special_tokens=False)["input_ids"]
    label_ids   = success_ids if success else failure_ids

    # 레이블 토큰 첫 등장 위치
    label_start = None
    for i in range(len(target_np) - len(label_ids) + 1):
        if list(target_np[i:i+len(label_ids)]) == label_ids:
            label_start = i
            break

    loss_mask = np.zeros(len(target_np), dtype=np.float32)
    if label_start is not None:
        loss_mask[label_start:] = 1.0   # SUCCESS/FAILURE ~ EOS 전체

    return target_ids, torch.FloatTensor(loss_mask)


# ─────────────────────────────────────────
# 학습 루프
# ─────────────────────────────────────────

def train_stage3(actor: ActorModel):
    print(f"\n{'='*60}")
    print(f"Stage 3: 성공/실패 예측 + 액션 생성 ({EPISODES_PER_TASK * len(TASK_IDS)} 에피소드)")
    print(f"  SUCCESS → SUCCESS 토큰 + 액션 7개 학습")
    print(f"  FAILURE → FAILURE 토큰만 학습 (텍스트 free)")
    print(f"{'='*60}\n")

    load_checkpoint(actor, LOAD_PATH)

    trainable = list(filter(lambda p: p.requires_grad, actor.parameters()))
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(trainable, lr=LEARNING_RATE)
        print("[optimizer] AdamW8bit")
    except ImportError:
        optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    # ── 데이터 수집 ────────────────────────────────────
    all_data     = []   # (img_np, planner_tokens, success, instruction, scene_texts)
    all_tasks    = []

    for task_id in TASK_IDS:
        env, instruction = make_env(task_id)
        print(f"\n[태스크 {task_id}] {instruction[:60]}")

        scene_texts = precompute_scene_texts(actor, env, instruction)
        rollout     = collect_rollout(actor, env, instruction, EPISODES_PER_TASK)
        env.close()

        for img_np, tokens, success in rollout:
            all_data.append((img_np, tokens, success, instruction, scene_texts))

    total     = len(all_data)
    n_success = sum(1 for *_, s, __, ___ in all_data if s)
    print(f"\n수집 완료: 총 {total}스텝 | SUCCESS={n_success} FAILURE={total-n_success}")

    # ── 학습 ───────────────────────────────────────────
    losses      = []
    global_step = 0

    for epoch in range(TRAIN_EPOCHS):
        np.random.shuffle(all_data)
        print(f"\n[Epoch {epoch+1}/{TRAIN_EPOCHS}]")

        for img_np, planner_tokens, success, instruction, scene_texts in all_data:
            image      = Image.fromarray(img_np)
            scene_text = scene_texts[np.random.randint(len(scene_texts))]

            try:
                cached_inputs, ihs = capture_ihs(
                    actor, image, instruction, planner_tokens
                )
            except Exception as e:
                print(f"  [IHS 캡처 실패] {e}")
                continue

            target_ids, loss_mask = make_target(
                actor, planner_tokens, scene_text, success
            )

            loss_val = sft_step(
                actor, optimizer, cached_inputs,
                planner_tokens, target_ids, loss_mask, ihs
            )
            losses.append(loss_val)
            global_step += 1

            if global_step % 100 == 0:
                valid = [l for l in losses[-100:] if l > 0]
                avg   = float(np.mean(valid)) if valid else 0.0
                print(f"  [Step {global_step:4d}] loss={avg:.4f}")

    # ── 저장 ───────────────────────────────────────────
    os.makedirs(SAVE_PATH, exist_ok=True)
    actor.smol.save_pretrained(SAVE_PATH)
    actor.processor.save_pretrained(SAVE_PATH)
    print(f"\n[Stage 3 완료] 저장: {SAVE_PATH}")
    print("다음: smol_train.py (GRPO)")


def main():
    print(f"[CUDA] {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '미설정')}")
    actor = ActorModel()
    actor.smol.gradient_checkpointing_enable()
    train_stage3(actor)
    actor.close()


if __name__ == "__main__":
    main()