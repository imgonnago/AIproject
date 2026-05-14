from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np

MODEL_PATH = "openvla/openvla-7b-finetuned-libero-10"

print("프로세서 로드 중...")
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("모델 로드 중... (시간이 소요될 수 있습니다)")
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
vla.eval()
print("모델 로드 완료!")

def get_action_tokens(
    image: Image.Image, # 로봇 이미지
    instruction: str # 명령 텍스트
) -> np.ndarray:
    # 프롬프트 포맷 (OpenVLA 공식 포맷)
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    # 입력 전처리
    inputs = processor(prompt, image).to("cpu", dtype=torch.bfloat16)

    # 모델 추론
    with torch.no_grad():
        generated_ids = vla.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    # 입력 token 제거 → action token만 추출
    input_len = inputs["input_ids"].shape[1]
    action_token_ids = generated_ids[0, input_len:input_len + 7]

    return action_token_ids.cpu().numpy().astype(int)

if __name__ == "__main__":
    test_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
    test_instruction = "pick up the black bowl on the left and place it on the plate"

    action_tokens = get_action_tokens(test_image, test_instruction)
    print(f"action token IDs: {action_tokens}")
    print(f"shape: {action_tokens.shape}")