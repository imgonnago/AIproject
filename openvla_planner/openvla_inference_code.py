from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import numpy as np
import zmq
import io

MODEL_PATH = "openvla/openvla-7b-finetuned-libero-spatial"

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


def run_server(port: int = 5555):
    """
    ZeroMQ REP 서버 실행.
    Actor(qwen_env)에서 이미지 + 텍스트를 받아
    action token을 생성해서 반환한다.

    :param port: 서버 포트 번호 (기본값 5555)
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP = 응답자
    socket.bind(f"tcp://*:{port}")
    print(f"[Planner 서버] 포트 {port} 대기 중...")

    while True:
        try:
            # 1. Actor에서 이미지 + 텍스트 수신
            data        = socket.recv_pyobj()
            image_bytes = data["image"]        # bytes
            instruction = data["instruction"]  # str
            print(f"[수신] instruction: {instruction}")

            # 2. bytes → PIL Image 변환
            image = Image.open(io.BytesIO(image_bytes))

            # 3. OpenVLA inference → action token 생성
            action_tokens = get_action_tokens(image, instruction)
            print(f"[생성] action tokens: {action_tokens}")

            # 4. action token → Actor로 반환
            socket.send_pyobj({
                "action_tokens": action_tokens,  # shape (7,), dtype int
                "status": "ok"
            })

        except Exception as e:
            print(f"[오류] {e}")
            # 오류 발생 시에도 반드시 send 해야 다음 요청 받을 수 있음
            socket.send_pyobj({
                "action_tokens": np.zeros(7, dtype=int),
                "status": "error",
                "error": str(e)
            })


if __name__ == "__main__":
    # 모델 로드 완료 후 서버 시작
    run_server(port=5555)