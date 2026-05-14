"""
ZeroMQ 클라이언트
- qwen_env에서 import해서 사용
- 이미지 + 텍스트를 OpenVLA 서버에 보내고 action token을 받는다.

사용 방법:
    from zmq_client import OpenVLAClient

    planner = OpenVLAClient()
    action_tokens = planner.get_action_tokens(image, instruction)
"""

import io
import zmq
import numpy as np
from PIL import Image


class OpenVLAClient:
    """
    OpenVLA ZeroMQ 클라이언트.
    training/grpo_train.py에서 Planner로 사용.
    """

    def __init__(self, host: str = "localhost", port: int = 5555):
        """
        서버에 연결.
        서버(openvla_server.py)가 먼저 실행되어 있어야 함.

        :param host: 서버 주소 (같은 컴퓨터면 localhost)
        :param port: 서버 포트 번호 (기본값 5555)
        """
        self.context = zmq.Context()

        # REQ 소켓 생성 (요청자 역할)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"[OpenVLA 클라이언트] 서버 연결 완료: {host}:{port}")

    def get_action_tokens(
        self,
        image: Image.Image,
        instruction: str
    ) -> np.ndarray:
        """
        이미지와 태스크 명령을 서버에 보내고
        action token ID 7개를 받아서 반환.

        :param image: PIL Image (LIBERO 환경 이미지)
        :param instruction: 태스크 명령 텍스트
        :return: shape (7,), dtype int, action token IDs
        """

        # 1. PIL Image → bytes 변환 (ZeroMQ 전송용)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # 2. 서버에 데이터 전송
        self.socket.send_pyobj({
            "image": image_bytes,
            "instruction": instruction
        })

        # 3. 서버 응답 대기 및 수신
        response = self.socket.recv_pyobj()

        # 오류 확인
        if response["status"] == "error":
            print(f"[경고] 서버 오류 발생: {response['error']}")
            print("[경고] zero action 반환")

        return response["action_tokens"]  # shape (7,), dtype int

    def close(self):
        """
        연결 종료.
        학습 끝나면 반드시 호출.
        """
        self.socket.close()
        self.context.term()
        print("[OpenVLA 클라이언트] 연결 종료")


# ─────────────────────────────────────────
# 통신 테스트 (단독 실행 시)
# ─────────────────────────────────────────

if __name__ == "__main__":
    """
    서버가 실행 중인 상태에서 테스트.

    터미널 1: conda activate openvla && python openvla/openvla_server.py
    터미널 2: conda activate qwen   && python qwen/zmq_client.py
    """

    planner = OpenVLAClient(host="localhost", port=5555)

    # 더미 이미지로 통신 테스트
    test_image       = Image.new("RGB", (224, 224), color=(128, 128, 128))
    test_instruction = "pick up the black bowl on the left and place it on the plate"

    print("\n=== ZeroMQ 통신 테스트 ===")
    print(f"instruction: {test_instruction}")

    action_tokens = planner.get_action_tokens(test_image, test_instruction)
    print(f"수신된 action tokens: {action_tokens}")
    print(f"shape: {action_tokens.shape}")  # (7,)

    planner.close()