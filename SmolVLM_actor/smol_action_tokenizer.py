import numpy as np
import torch
import torch.nn as nn
from smol_projection_layer import Projection


class ActorActionTokenizer:
    """
    Actor 모델의 action token 처리 담당 클래스.

    역할 (Paradigm B - token ID 기반):
        [초기화 - 1회]
            add_tokenizer_vocab()      : SmolVLM2 vocab에 <action_0>~<action_255> 추가
            resize_embeddings()        : SmolVLM2 임베딩 테이블 확장
            init_action_embeddings()   : 추가된 256개 토큰을 OpenVLA 임베딩으로 초기화
                                         (OpenVLA embedding → Projection → SmolVLM2 embedding 공간)
                                         목적: 랜덤 초기화 대신 로봇 동작 의미론이 담긴 시작점 제공

        [매 스텝 - 프롬프트 구성용]
            openvla_ids_to_bin_indices()  : OpenVLA raw token ID → bin index (0~255)
            bin_indices_to_continuous()   : bin index → 연속값 (-1~+1), 프롬프트 텍스트 표시용

        [출력 파싱]
            decode_token_ids_to_actions() : SmolVLM2 출력 action token ID → 연속값 복원

    ※ 제거된 함수:
        embed_action_tokens() - Paradigm A dead code (inputs_embeds 방식, 미사용)
        forward()             - Paradigm A dead code (임베딩 concat, 미사용)
        실제 구현은 input_ids에 action token ID를 이어붙이는 방식 (Paradigm B)
    """

    def __init__(self, processor, smol_model, projection, OPENVLA_VOCAB_SIZE=32000):
        self.bins        = np.linspace(-1, 1, 256)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.min_action  = -1
        self.max_action  = 1
        self.processor   = processor
        self.smol_model  = smol_model
        self.projection  = projection
        self.OPENVLA_VOCAB_SIZE = OPENVLA_VOCAB_SIZE

        # ── OpenVLA 임베딩 테이블 로드 (frozen) ──────────────────────────────
        # init_action_embeddings()에서 1회만 사용.
        # 역할: OpenVLA가 학습한 action 의미론(semantics)을
        #        Projection을 통해 SmolVLM2 임베딩 공간으로 이식하는 '번역기'
        action_embed_weights = torch.load(
            "assets/openvla_action_embeddings.pt",
            weights_only=False
        )
        if action_embed_weights.shape[0] != 256:
            action_embed_weights = action_embed_weights[-256:]

        # shape: (256, 4096), freeze=True → 초기화 후 gradient 없음
        self.openvla_embedding = nn.Embedding.from_pretrained(
            action_embed_weights,
            freeze=True
        ).to("cuda")


    # ─────────────────────────────────────────
    # 초기화 함수 (한 번만 호출)
    # ─────────────────────────────────────────

    def add_tokenizer_vocab(self, n_bins: int = 256):
        """
        SmolVLM2 tokenizer vocab에 action token 256개 추가.
        <action_0> ~ <action_255> special token 추가.

        SmolVLM2 기본 vocab_size: 49152
        추가 후: 49152 + 256 = 49408
        """
        action_tokens = [f"<action_{i}>" for i in range(n_bins)]
        num_added = self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": action_tokens}
        )
        print(f"[add_tokenizer_vocab] 추가된 token 수: {num_added}")
        print(f"[add_tokenizer_vocab] tokenizer 새 크기: {len(self.processor.tokenizer)}")
        return self.processor.tokenizer

    def resize_embeddings(self):
        """
        SmolVLM2 임베딩 테이블 크기를 tokenizer vocab 크기에 맞게 확장.
        새로 추가된 256개 행은 init_action_embeddings()에서 덮어씀.
        """
        self.smol_model.resize_token_embeddings(len(self.processor.tokenizer))
        print(f"[resize_embeddings] 임베딩 테이블 크기: {self.smol_model.get_input_embeddings().weight.shape}")
        return self.smol_model

    def init_action_embeddings(self, openvla_embed_weights: torch.Tensor):
        """
        추가된 action token 256개를 OpenVLA 임베딩으로 초기화 (1회).

        흐름:
            OpenVLA embedding (256, 4096)  [frozen openvla_embedding table에서]
            → Projection Layer (4096 → 960)
            → SmolVLM2 임베딩 테이블 마지막 256행 교체

        효과:
            <action_128> 토큰은 처음부터 "중립적 동작(~0.0)"에 해당하는
            의미있는 벡터로 초기화됨 → 랜덤 초기화 대비 GRPO 수렴 가속

        이후 GRPO 학습을 통해 이 임베딩들이 계속 업데이트됨.
        """
        if openvla_embed_weights.shape[0] != 256:
            openvla_embed_weights = openvla_embed_weights[-256:]

        with torch.no_grad():
            # OpenVLA embedding → Projection → SmolVLM2 공간 (256, 960)
            init_weights = self.projection(
                openvla_embed_weights.to("cuda").float()
            )

            embed_layer = self.smol_model.get_input_embeddings()
            embed_layer.weight.data[-256:] = init_weights.to(torch.bfloat16)

        print("[init_action_embeddings] Action embeddings 초기화 완료!")

    def setup(self, openvla_embed_weights: torch.Tensor):
        """
        ActorActionTokenizer 초기화를 순서대로 한 번에 처리.
        ActorModel.__init__에서 한 번만 호출.
        """
        self.add_tokenizer_vocab()
        self.resize_embeddings()
        self.init_action_embeddings(openvla_embed_weights)
        print("[setup] ActorActionTokenizer 초기화 완료!")


    # ─────────────────────────────────────────
    # 변환 유틸리티 (매 스텝, 프롬프트 구성용)
    # ─────────────────────────────────────────

    def openvla_ids_to_bin_indices(self, openvla_token_ids: np.ndarray) -> np.ndarray:
        """
        OpenVLA raw token ID → bin index (0~255) 변환.

        OpenVLA 인코딩 방식:
            token_id = vocab_size - np.digitize(action)
            → 역산: bin_index = vocab_size - token_id

        예시:
            OpenVLA vocab_size = 32000
            token_id = 31872  →  bin_index = 32000 - 31872 = 128  (중립값 ~0.0)
            token_id = 31744  →  bin_index = 256  →  clamp → 255  (최대값 +1.0)
            token_id = 31999  →  bin_index = 1                     (최소값 ~-1.0)

        :param openvla_token_ids: shape (7,), OpenVLA vocab 기준 raw token IDs
        :return: shape (7,), bin indices (0~255)
        """
        bin_indices = self.OPENVLA_VOCAB_SIZE - np.array(openvla_token_ids)
        return np.clip(bin_indices, 0, 255).astype(int)

    def bin_indices_to_continuous(self, bin_indices: np.ndarray) -> np.ndarray:
        """
        bin index → 연속값 (-1.0 ~ +1.0) 변환.

        프롬프트 텍스트에 실제 값을 표시하여 모델이
        "이 값이 왜 잘못됐는지" 텍스트로 추론하게 함.

        :param bin_indices: shape (7,), 0~255
        :return: shape (7,), 연속값 action vector
        """
        clipped = np.clip(bin_indices, 0, len(self.bin_centers) - 1)
        return self.bin_centers[clipped]

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        SmolVLM2 출력 action token ID → 연속값 복원.

        SmolVLM2 action token 범위:
            vocab_size after add: 49408
            action token start:   49408 - 256 = 49152
            <action_0>  → ID 49152 → bin_index 0  → bin_centers[0]  ≈ -0.992
            <action_128>→ ID 49280 → bin_index 128 → bin_centers[128] ≈ 0.000
            <action_255>→ ID 49407 → bin_index 255 → bin_centers[254] ≈ +0.992

        :param action_token_ids: shape (7,), SmolVLM2 출력 token IDs (49152~49407)
        :return: shape (7,), 연속값 action vector (-1~+1)
        """
        smol_action_token_start = len(self.processor.tokenizer) - 256
        discretized_actions = action_token_ids - smol_action_token_start  # 0~255
        discretized_actions = np.clip(
            discretized_actions,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1
        )
        return self.bin_centers[discretized_actions]