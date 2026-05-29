"""
smol_sft_stage1.py — Stage 1: 텍스트 다양성 확인 (학습 없음)

[목적]
    reset_action_lm_head() 적용 후 실제 LIBERO 이미지를 보고
    모델이 다양한 텍스트를 자연스럽게 생성하는지 확인.

[핵심]
    reset_action_lm_head(): lm_head action token 편향 제거
    disable_adapter(): LoRA 비활성화 → 베이스 SmolVLM2 사용
    apply_chat_template: SmolVLM2 올바른 processor 호출

[판단 기준]
    다양한 이미지 설명 텍스트 → Stage 2 진행
    여전히 action 토큰 / 빈 텍스트 → 문제 확인 필요

실행:
    python train/smol_sft_stage1.py
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../SmolVLM_actor"))

import torch
import numpy as np
from PIL import Image

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from SmolVLM_actor.smol_actor_model import ActorModel

CHECK_STEPS = 20
TASK_SUITE  = "libero_10"
TASK_IDS    = [0, 1, 2]   # 다양한 장면 확인
IMG_HEIGHT  = 224
IMG_WIDTH   = 224


# ─────────────────────────────────────────
# lm_head action token 편향 제거
# ─────────────────────────────────────────

def reset_action_lm_head(actor: ActorModel):
    lm_head  = actor.smol.get_output_embeddings()
    regular  = lm_head.weight.data[:-256]
    mean = regular.mean(dim=0, keepdim=True)
    std  = regular.std(dim=0, keepdim=True).clamp(min=1e-6)
    with torch.no_grad():
        new_w = torch.normal(mean.expand(256, -1), std.expand(256, -1))
        lm_head.weight.data[-256:] = new_w.to(lm_head.weight.dtype)
    before = regular.norm(dim=1).mean().item()
    after  = lm_head.weight.data[-256:].norm(dim=1).mean().item()
    print(f"[reset_action_lm_head] 일반 토큰 norm: {before:.3f} | action 토큰 norm: {after:.3f}")


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
# 텍스트 생성 (disable_adapter + apply_chat_template)
# ─────────────────────────────────────────

def generate_text(actor: ActorModel, image: Image.Image, instruction: str) -> str:
    """
    lm_head 편향 제거 + LoRA 비활성화 상태에서
    실제 이미지를 보고 텍스트 생성.
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Briefly describe the scene for: {instruction}"}
        ]
    }]

    inputs = actor.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    try:
        with torch.no_grad():
            with actor.smol.disable_adapter():
                out = actor.smol.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=True,
                    temperature=0.7,
                )
        prompt_len = inputs["input_ids"].shape[1]
        text = actor.processor.decode(
            out[0][prompt_len:], skip_special_tokens=True
        ).strip()
        del out
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")
        text = ""

    del inputs
    torch.cuda.empty_cache()
    return text


# ─────────────────────────────────────────
# 다양성 지표
# ─────────────────────────────────────────

def diversity_score(texts: list) -> dict:
    non_empty = [t for t in texts if t and t.strip()]
    if not non_empty:
        return {"non_empty": 0, "unique_ratio": 0.0, "avg_words": 0, "vocab_size": 0}
    all_words = " ".join(non_empty).lower().split()
    return {
        "non_empty":    len(non_empty),
        "unique_ratio": len(set(non_empty)) / len(non_empty),
        "avg_words":    len(all_words) / len(non_empty),
        "vocab_size":   len(set(all_words)),
    }


# ─────────────────────────────────────────
# Stage 1 확인
# ─────────────────────────────────────────

def check_stage1(actor: ActorModel):
    print(f"\n{'='*60}")
    print(f"Stage 1: 텍스트 다양성 확인 (학습 없음)")
    print(f"{'='*60}\n")

    # lm_head 편향 제거 (핵심)
    reset_action_lm_head(actor)

    all_texts = []

    for task_id in TASK_IDS:
        env, task_name = make_env(task_id)
        print(f"\n[태스크 {task_id}] {task_name[:60]}")
        obs = env.reset()

        for step in range(CHECK_STEPS // len(TASK_IDS)):
            image = get_image(obs)
            text  = generate_text(actor, image, task_name)
            all_texts.append(text)

            status = "✓" if text and len(text.split()) >= 3 else "✗"
            print(f"  [{status}] {text[:100] if text else '(없음)'}")

            obs, _, done, _ = env.step(np.zeros(7))
            if done:
                obs = env.reset()

        env.close()

    # 다양성 리포트
    score = diversity_score(all_texts)
    print(f"\n{'='*60}")
    print(f"다양성 리포트 ({CHECK_STEPS}개 이미지)")
    print(f"  텍스트 생성 성공: {score['non_empty']}/{CHECK_STEPS}")
    print(f"  고유 텍스트 비율: {score['unique_ratio']:.2f}  (1.0 = 전부 다름)")
    print(f"  평균 단어 수:     {score['avg_words']:.1f}")
    print(f"  고유 단어 수:     {score['vocab_size']}")

    if score["non_empty"] == 0:
        print("\n❌ 텍스트 없음 — 문제 확인 필요")
    elif score["unique_ratio"] < 0.5 or score["avg_words"] < 3:
        print("\n⚠️  다양성 부족 — Stage 2 진행 전 확인 권장")
    else:
        print("\n✅ 다양성 양호 — Stage 2 진행 가능")

    print("\n다음: smol_sft_stage2.py")


def main():
    actor = ActorModel()
    check_stage1(actor)
    actor.close()


if __name__ == "__main__":
    main()