# openvla_env에서 실행
import torch
import os
from transformers import AutoModelForVision2Seq

os.makedirs("assets", exist_ok=True)

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b-finetuned-libero-10",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

# 마지막 256개 행만 추출
embed_weights = model.language_model.model.embed_tokens.weight[-256:].detach().clone().float()
torch.save(embed_weights, "assets/openvla_action_embeddings.pt")
print(f"저장 완료: shape = {embed_weights.shape}")  # (256, 4096)
