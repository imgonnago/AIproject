"""
smol_action_tokenizer.py

Actor 모델의 action token 처리 담당 클래스.

__init__ : Openvla 임베딩 테이블 로드 및 변수 초기화

add_tokenizer_vocab : SmolVLM 500M tokenizer에 openvla action token 256개 추가

resize_embeddings : SmolVLM 500M 임베딩 테이블 크기 확장

init_action_embeddings : 추가된 action token을 OpenVLA 임베딩으로 초기화

setup : 위 3단계 초기화 함수 순서대로 실행

embed_action_tokens : 매 스텝 OpenVLA token ID → SmolVLM2 입력 임베딩 변환 (Paradigm A 핵심)

openvla_ids_to_bin_indices : OpenVLA raw token ID → bin index (0~255) 변환

bin_indices_to_continuous : bin index → 연속값 (-1.0 ~ +1.0) 변환

decode_token_ids_to_actions : SmolVLM2 출력 <action_N> token ID → 연속값 복원
"""

import numpy as np
import torch
import torch.nn as nn
from smol_projection_layer import Projection


class ActorActionTokenizer:

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
        # Paradigm A의 핵심: OpenVLA가 학습한 action 의미론이 담긴 (256, 4096) 벡터들
        # embed_action_tokens()에서 매 스텝 호출됨
        # Projection과 함께 "OpenVLA action → SmolVLM2 임베딩 공간" 번역을 담당
        action_embed_weights = torch.load(
            "assets/openvla_action_embeddings.pt",
            weights_only=False
        )
        if action_embed_weights.shape[0] != 256:
            action_embed_weights = action_embed_weights[-256:]

        # shape: (256, 4096), freeze=True → OpenVLA 의미론 고정, gradient 없음
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
        추가 후: 49152 + 256 = 49408 (실제로는 49280 + 256 = 49536일 수도 있음)
        → len(self.processor.tokenizer)로 항상 동적으로 계산

        이 토큰들은 SmolVLM2의 출력 어휘로 사용됨:
            모델이 [ACTION] 태그 안에 <action_N> 토큰들을 생성
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
        추가된 action token 256개를 OpenVLA 임베딩으로 초기화.
        프로젝션 레이어를 통해 openvla 임베딩 차원을 smolvlm 임베딩 차원으로 변환하여 초기화.
        초기화는 한 번만 수행되고, 이후 학습 과정에서 Projection 레이어가 "OpenVLA 공간 → SmolVLM2 공간" 번역을 점점 더 잘 학습하게 됨.
        """
        smol_action_start = len(self.processor.tokenizer) - 256
        
        # [수정] 크리티컬 1번 방어: openvla_embed_weights의 형태 검사
        # openvla_embed_weights가 256행이 아니면 projection 출력이 잘못된 shape이 되어
        # in_emb.weight.data 대입 시 dimension mismatch 에러가 발생하는 것을 방지합니다.
        if openvla_embed_weights.shape[0] != 256:
            openvla_embed_weights = openvla_embed_weights[-256:]

        with torch.no_grad():
            init_weights = self.projection(openvla_embed_weights.to("cuda").float())

            # 1. Input Embeddings 초기화
            in_emb = self.smol_model.get_input_embeddings()
            in_emb.weight.data[smol_action_start:] = init_weights.to(torch.bfloat16)

            # 2. LM Head 초기화 (Weight Tying이 False일 때 필수)
            out_emb = self.smol_model.get_output_embeddings()
            if out_emb is not None:
                out_emb.weight.data[smol_action_start:] = init_weights.to(torch.bfloat16)
        
        # [수정] 크리티컬 2번 방어: lm_head requires_grad 명시적 허용
        # Weight tying이 없을 경우 lm_head가 얼어있으면 학습이 진행되어도 출력 확률이 오르지 않으므로
        # 명시적으로 gradient 계산을 허용해줍니다.
        if out_emb is not None:
            out_emb.weight.requires_grad_(True)
            print("[init_action_embeddings] lm_head gradient 허용 완료!")

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
    # [Paradigm A 핵심] 매 스텝 임베딩 변환
    # ─────────────────────────────────────────

    def embed_action_tokens(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        """
        OpenVLA action token ID → SmolVLM2 입력 임베딩 변환.

        흐름:
            OpenVLA raw token IDs (31744~31999)
            → bin_indices (0~255): vocab_size - token_id
            → openvla_embedding (frozen, 256, 4096): 로봇 동작 의미론 담긴 벡터
            → Projection layer (4096→960, 학습가능): SmolVLM2 공간으로 번역
            → unsqueeze(0) → (1, 7, 960)

        학습 중 gradient 흐름:
            GRPO loss → logits → transformer → inputs_embeds → action_embeds → Projection
            → Projection이 "OpenVLA 공간 → SmolVLM2 공간" 번역을 점점 잘 배움

        :param action_token_ids: shape (7,), OpenVLA raw token IDs
        :return: shape (1, 7, 960), bfloat16
        """
        # raw OpenVLA token ID → bin index (0~255)
        bin_indices = self.OPENVLA_VOCAB_SIZE - action_token_ids -1
        bin_indices = torch.clamp(bin_indices, min=0, max=255)

        # OpenVLA 임베딩 테이블에서 lookup (frozen): (7, 4096)
        openvla_embeds = self.openvla_embedding(
            bin_indices.clone().detach().to(dtype=torch.long).to("cuda")
        )

        # Projection: (7, 4096) → (7, 960), 학습 가능
        smol_embeds = self.projection(openvla_embeds.float())

        return smol_embeds.unsqueeze(0).to(torch.bfloat16)  # (1, 7, 960)


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
        bin_indices = self.OPENVLA_VOCAB_SIZE - np.array(openvla_token_ids) -1
        return np.clip(bin_indices, 0, 255).astype(int)

    def bin_indices_to_continuous(self, bin_indices: np.ndarray) -> np.ndarray:
        """
        bin index → 연속값 (-1.0 ~ +1.0) 변환.

        :param bin_indices: shape (7,), 0~255
        :return: shape (7,), 연속값 action vector
        """
        clipped = np.clip(bin_indices, 0, len(self.bin_centers) - 1)
        return self.bin_centers[clipped]

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        SmolVLM2 출력 <action_N> token ID → 연속값 복원.

        SmolVLM2 action token 범위:
            vocab_size after add: 49536 (또는 49408, 동적으로 계산)
            action token start:   len(tokenizer) - 256
            <action_0>  → ID smol_action_start+0   → bin_index 0   → bin_centers[0]  ≈ -0.992
            <action_128>→ ID smol_action_start+128  → bin_index 128 → bin_centers[128] ≈ 0.000
            <action_255>→ ID smol_action_start+255  → bin_index 255 → bin_centers[254] ≈ +0.992

        :param action_token_ids: shape (7,), SmolVLM2 출력 token IDs
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