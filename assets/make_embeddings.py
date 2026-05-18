import os
import torch
from transformers import AutoModelForVision2Seq

print("OpenVLA-7B Long 버전 모델 로드 중 (임베딩 추출용)...")

model_id = "openvla/openvla-7b-finetuned-libero-10"

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

vla_embeddings = model.get_input_embeddings().weight.data
print(f"추출된 Long 버전 임베딩 셰이프: {vla_embeddings.shape}")

output_path = "openvla_action_embeddings.pt"
torch.save(vla_embeddings, output_path)
print(f"✅ 성공적으로 {output_path} 파일을 생성했습니다!")
