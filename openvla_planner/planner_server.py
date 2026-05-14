"""
OpenVLA ZeroMQ 서버
- openvla_env에서 실행
- qwen_env(클라이언트)에서 이미지 + 텍스트를 받아
  action token을 생성해서 반환한다.

실행 방법:
    conda activate openvla
    python openvla/openvla_server.py
"""

import io
import zmq
import numpy as np
from PIL import Image
from openvla_inference_code import get_action_tokens


def run_server(port: int = 5555):
    """
    ZeroMQ REP 서버 실행.
    클라이언트 요청을 받아 OpenVLA inference 후 action token 반환.

    :param port: 서버 포트 번호 (기본값 5555)
    """

    # ZeroMQ 컨텍스트 생성
    context = zmq.Context()

    # REP 소켓 생성 (응답자 역할)
    socket = context.socket(zmq.REP)

    # 포트 열기
    socket.bind(f"tcp://*:{port}")
    print(f"[OpenVLA 서버] 포트 {port} 대기 중...")

    while True:
        try:
            # 1. 클라이언트에서 데이터 수신
            data        = socket.recv_pyobj()
            image_bytes = data["image"]        # bytes
            instruction = data["instruction"]  # str
            print(f"[수신] instruction: {instruction}")

            # 2. bytes → PIL Image 변환
            image = Image.open(io.BytesIO(image_bytes))

            # 3. OpenVLA inference → action token 생성
            action_tokens = get_action_tokens(image, instruction)
            print(f"[생성] action tokens: {action_tokens}")

            # 4. action token 반환
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
    run_server(port=5555)