import numpy as np
import torch
import torch.nn as nn
from projection_layer import Projection

OPENVLA_VOCAB_SIZE = 32000

class ActorActionTokenizer:
    def __init__(self, processor, qwen_model, projection):
        # ActorActionTokenizer __init__에 추가 필요
        self.bins = np.linspace(-1, 1, 256)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.min_action = -1
        self.max_action = 1
        self.processor = processor
        self.qwen_model = qwen_model
        self.projection = projection

        action_embed_weights = torch.load(
        "assets/openvla_action_embeddings.pt"
    )  # shape (256, 4096)
        self.openvla_embedding = nn.Embedding.from_pretrained(
        action_embed_weights,
        freeze=True  # 학습 안됨
    ).to("cuda")

    def add_tokenizer_vocab(self, n_bins: int = 256):
        """
        Adds `n_bins` new tokens to the tokenizer's vocabulary for discretized action representation.

        NOTE =>> This method should be called after initializing the ActionTokenizer and before training/inference.

        :param tokenizer: The base LLM/VLM tokenizer to which we want to add new tokens.
        :param n_bins: The number of discrete bins (and thus new tokens) to add for action representation.
        """
        action_tokens = [f"<action_{i}>" for i in range(n_bins)]
        self.processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": action_tokens}
        )   
        return self.processor.tokenizer
        
            
    def resize_embeddings(self):
        """
        Resizes the model's token embeddings to accommodate the new tokens added to the tokenizer.

        NOTE =>> This method should be called after `add_tokenizer_vocab` to ensure the model can handle the new tokens.

        :param model: The LLM/VLM model whose token embeddings need to be resized.
        :param tokenizer: The tokenizer that has been updated with new tokens.
        """
        self.qwen_model.resize_token_embeddings(len(self.processor.tokenizer))
        return self.qwen_model
        
    def init_action_embeddings(self, openvla_embed_weights: torch.Tensor):
        """
        새로 추가된 action token 256개를 OpenVLA 임베딩으로 초기화.
        랜덤 초기화 대신 의미 있는 초기값으로 학습 수렴 속도 향상.

        :param openvla_embed_weights: shape (256, 4096), OpenVLA에서 추출한 임베딩
        """
        with torch.no_grad():
            # Projection Layer로 4096 → 2048 변환
            init_weights = self.projection(
                openvla_embed_weights.to("cuda").float()
            )  # (256, 2048)

            # 새로 추가된 마지막 256개 행만 교체
            self.qwen_model.get_input_embeddings().weight[-256:] = (
                init_weights.to(torch.bfloat16)
            )
    
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = OPENVLA_VOCAB_SIZE - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]
    

    def encode_action_tokens(self, action_token_ids):
        """
        Discrete action token IDs → OpenVLA 임베딩 → Projection → Qwen 공간

        :param action_token_ids: shape (7,), dtype int, OpenVLA에서 생성된 action token ID 7개
        :return: shape (7, 2048), dtype float, Qwen 모델에서 사용할 수 있는 action 임베딩
        """
        # 1. Token IDs → OpenVLA 임베딩
        bin_indices = OPENVLA_VOCAB_SIZE - action_token_ids  # 0~255
        openvla_embeds = self.openvla_embedding(
            torch.tensor(bin_indices, dtype=torch.long).to("cuda")
        )  # (7, 4096)

        # 2. Projection Layer로 4096 → 2048 변환
        qwen_embeds = self.projection(openvla_embeds.float())  # (7, 2048)

        return qwen_embeds.to(torch.bfloat16)
    
    def concat_embeddings(self, text_image_embeds, action_embeds):
        """
        텍스트/이미지 임베딩 + action 임베딩 concat

        :param text_image_embeds: shape (seq_len, 2048), 텍스트/이미지 임베딩
        :param action_embeds: shape (7, 2048), action 임베딩
        :return: shape (seq_len + 7, 2048), concat된 임베딩
        """
        return torch.cat([text_image_embeds, action_embeds], dim=1)