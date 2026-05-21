import numpy as np
import torch
import torch.nn as nn
from smol_projection_layer import Projection


class ActorActionTokenizer:
    """
    Actor 모델의 action token 처리를 담당하는 클래스.

    변경사항 (Qwen → SmolVLM2-500M):
        - SMOL_DIM: 2048 → 960 (SmolLM2-360M hidden_size)
        - openvla_embedding: 4096차원 → Projection → 960차원
        - decode_token_ids_to_actions: Qwen vocab 기준 그대로 유지
          (SmolVLM2 vocab size는 49152 + 추가 token)

    주요 함수:
        setup()                    ← 초기화 (한 번만)
        forward()                  ← 매 스텝 (embed + concat)
        decode_token_ids_to_actions() ← 출력 복원
    """

    def __init__(self, processor, smol_model, projection, OPENVLA_VOCAB_SIZE=32000):
        self.bins        = np.linspace(-1, 1, 256)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.min_action  = -1
        self.max_action  = 1
        self.processor   = processor
        self.smol_model  = smol_model      # SmolVLM2 모델
        self.projection  = projection      # Projection Layer (4096 → 960)
        self.OPENVLA_VOCAB_SIZE = OPENVLA_VOCAB_SIZE

        # OpenVLA 임베딩 테이블 로드 (frozen)
        action_embed_weights = torch.load(
            "assets/openvla_action_embeddings.pt",
            weights_only=False
        )
        # 전체 임베딩이 저장된 경우 마지막 256개만 사용
        if action_embed_weights.shape[0] != 256:
            action_embed_weights = action_embed_weights[-256:]

        # shape: (256, 4096) frozen
        self.openvla_embedding = nn.Embedding.from_pretrained(
            action_embed_weights,
            freeze=True
        ).to("cuda")


    # ─────────────────────────────────────────
    # 초기화 함수들
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
        SmolVLM2 임베딩 테이블 크기를 tokenizer vocab 크기에 맞게 조정.
        """
        self.smol_model.resize_token_embeddings(len(self.processor.tokenizer))
        print(f"[resize_embeddings] 임베딩 테이블 크기: {self.smol_model.get_input_embeddings().weight.shape}")
        return self.smol_model

    def init_action_embeddings(self, openvla_embed_weights: torch.Tensor):
        """
        새로 추가된 action token 256개를 OpenVLA 임베딩으로 초기화.

        흐름:
            OpenVLA 임베딩 (256, 4096)
            → Projection Layer (4096 → 960)   ← SmolLM2-360M hidden_size
            → SmolVLM2 임베딩 테이블 마지막 256개 행 교체
        """
        if openvla_embed_weights.shape[0] != 256:
            openvla_embed_weights = openvla_embed_weights[-256:]

        with torch.no_grad():
            init_weights = self.projection(
                openvla_embed_weights.to("cuda").float()
            )  # (256, 960)

            embed_layer = self.smol_model.get_input_embeddings()
            embed_layer.weight.data[-256:] = init_weights.to(torch.bfloat16)

        print("[init_action_embeddings] Action embeddings 초기화 완료!")

    def setup(self, openvla_embed_weights: torch.Tensor):
        """
        ActorActionTokenizer 초기화를 한 번에 처리.
        actor_model.__init__에서 한 번만 호출.

        순서:
            1. add_tokenizer_vocab()
            2. resize_embeddings()
            3. init_action_embeddings()
        """
        self.add_tokenizer_vocab()
        self.resize_embeddings()
        self.init_action_embeddings(openvla_embed_weights)
        print("[setup] ActorActionTokenizer 초기화 완료!")


    # ─────────────────────────────────────────
    # 매 스텝 호출 함수들
    # ─────────────────────────────────────────

    def embed_action_tokens(self, action_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Planner(OpenVLA) action token ID → SmolVLM2 입력 임베딩 변환.

        흐름:
            OpenVLA token IDs (31744~31999)
            → bin_indices (0~255) = OPENVLA_VOCAB_SIZE - token_id
            → OpenVLA 임베딩 테이블 (frozen) → (7, 4096)
            → Projection Layer → (7, 960)     ← SmolLM2-360M hidden_size
            → unsqueeze(0) → (1, 7, 960)

        :param action_token_ids: shape (7,), OpenVLA action token IDs
        :return: shape (1, 7, 960)
        """
        bin_indices = self.OPENVLA_VOCAB_SIZE - action_token_ids
        bin_indices = torch.clamp(bin_indices, min=0, max=255)

        openvla_embeds = self.openvla_embedding(
            bin_indices.clone().detach().to(dtype=torch.long).to("cuda")
        )  # (7, 4096)

        smol_embeds = self.projection(openvla_embeds.float())  # (7, 960)

        return smol_embeds.unsqueeze(0).to(torch.bfloat16)  # (1, 7, 960)

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        SmolVLM2 출력 action token ID → 연속값 action vector 복원.

        SmolVLM2 action token 범위:
            기본 vocab_size: 49152
            추가 후: 49408
            action token start: 49408 - 256 = 49152

        :param action_token_ids: shape (7,), SmolVLM2 출력 action token IDs
        :return: shape (7,), 연속값 action vector
        """
        smol_action_token_start = len(self.processor.tokenizer) - 256
        discretized_actions = action_token_ids - smol_action_token_start  # 0~255
        discretized_actions = np.clip(
            discretized_actions,
            a_min=0,
            a_max=self.bin_centers.shape[0] - 1
        )
        return self.bin_centers[discretized_actions]

    def concat_embeddings(
        self,
        text_image_embeds: torch.Tensor,
        action_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        text/image 임베딩 + action 임베딩 concat.

        :param text_image_embeds: (batch, seq_len, 960)
        :param action_embeds: (batch, 7, 960)
        :return: (batch, seq_len+7, 960)
        """
        return torch.cat([text_image_embeds, action_embeds], dim=1)

    def forward(
        self,
        action_token_ids: torch.Tensor,
        text_image_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        매 추론 스텝 메인 함수.
        Planner action token → 임베딩 → SmolVLM2 입력과 concat.

        :param action_token_ids: shape (7,), Planner action token IDs
        :param text_image_embeds: shape (batch, seq_len, 960)
        :return: shape (batch, seq_len+7, 960)
        """
        action_embeds = self.embed_action_tokens(action_token_ids)  # (1, 7, 960)
        return self.concat_embeddings(text_image_embeds, action_embeds)