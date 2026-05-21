import numpy as np
import torch
import torch.nn as nn
from projection_layer import Projection

#openvla LLM vocab size

class ActorActionTokenizer:
    def __init__(self, processor, qwen_model, projection, OPENVLA_VOCAB_SIZE = 32000):
        self.bins = np.linspace(-1, 1, 256) # 256 bins between -1 and 1
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0 # bin centers for decoding
        self.min_action = -1 # action 값의 최소값 (클리핑, bin 간격의 하한 설정)
        self.max_action = 1 # action 값의 최대값 (클리핑, bin 간격의 상한 설정)
        self.processor = processor #qwen 모델의 processor (tokenizer + feature extractor)
        self.qwen_model = qwen_model # Qwen 모델 (action token 임베딩 레이어 resize 필요)
        self.projection = projection # OpenVLA 임베딩 → Qwen 모델 입력 임베딩으로 변환하는 Projection Layer (LayerNorm + MLP)
        self.OPENVLA_VOCAB_SIZE = OPENVLA_VOCAB_SIZE

        # OpenVLA에서 추출한 action token 임베딩으로 초기화된 임베딩 레이어 (새로 추가된 256개 토큰)
        action_embed_weights = torch.load(
        "assets/openvla_action_embeddings.pt",
        weights_only=False
        )  
        # shape (256, 4096)
        self.openvla_embedding = nn.Embedding.from_pretrained(
        action_embed_weights,
        freeze=True  # 학습 안됨
        ).to("cuda")


    def add_tokenizer_vocab(self, n_bins: int = 256):
        """
        qwen 모델의 tokenizer vocab에 action token 256개 추가.
        여기선 실제 토큰이 들어있는건 아니고, 단지 토크나이저가 인식할 수 있는 새로운 토큰 ID 256개를 추가하는 것임.
        이 토큰 ID들은 모델의 임베딩 레이어에서 OpenVLA 임베딩으로 초기화될 예정.
        """
        action_tokens = [f"<action_{i}>" for i in range(n_bins)]
        num_added = self.processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": action_tokens}
        )
        print(len(self.processor.tokenizer))
        # 현재 special tokens 확인
        print(f"추가된 token 수: {num_added}")
        print(f"tokenizer 새 크기: {len(self.processor.tokenizer)}")
        print(self.processor.tokenizer.special_tokens_map)
        print(len(self.processor.tokenizer))
          
        return self.processor.tokenizer
        
            
    def resize_embeddings(self):
        """
        qwen 모델의 임베딩 레이어 크기를 tokenizer vocab 크기에 맞게 조정.
        새로 추가된 256개 토큰의 임베딩은 OpenVLA에서 추출한 임베딩으로 초기화될 예정.
        """
        self.qwen_model.resize_token_embeddings(len(self.processor.tokenizer))
        print(len(self.qwen_model.get_input_embeddings().weight))
        print(f"[resize_embeddings] 임베딩 테이블 크기: {self.qwen_model.get_input_embeddings().weight.shape}")
        return self.qwen_model
        
    def init_action_embeddings(self, openvla_embed_weights: torch.Tensor):
        """
        새로 추가된 action token 256개를 OpenVLA 임베딩으로 초기화.
        랜덤 초기화 대신 의미있는 초기값으로 시작해서 학습 수렴 속도 향상.
 
        흐름:
            OpenVLA 임베딩 (256, 4096)
            → Projection Layer (4096 → 2048)
            → Qwen 임베딩 테이블 마지막 256개 행에 덮어씀
 
        :param openvla_embed_weights: shape (256, 4096) 또는 전체 임베딩 테이블
        """
        # 전체 임베딩이 들어온 경우 마지막 256개 행만 사용
        if openvla_embed_weights.shape[0] != 256:
            openvla_embed_weights = openvla_embed_weights[-256:]
 
        with torch.no_grad():
            # Projection Layer로 4096 → 2048 변환
            init_weights = self.projection(
                openvla_embed_weights.to("cuda").float()
            )  # (256, 2048)
 
            # Qwen 임베딩 테이블 마지막 256개 행 교체
            embed_layer = self.qwen_model.get_input_embeddings()
            embed_layer.weight.data[-256:] = init_weights.to(torch.bfloat16)
 
        print("[init_action_embeddings] Action embeddings 초기화 완료!")

    
    def embed_action_tokens(self, action_token_ids):
        """
        action token ID 7개 → OpenVLA 임베딩 → Projection Layer로 4096 → 2048 변환 → Qwen 모델 입력 임베딩
        qwen 모델에서 생성된 action token ID는 openvla에서 생성된 token ID와 동일한 방식으로 binning되어 있고
        같은 임베딩 레이어를 공유하기 때문에, action token ID → OpenVLA 임베딩 → Projection Layer로 4096 → 2048 변환 → Qwen 모델 입력 임베딩으로 변환하는 과정은 
        openvla의 tokenizer + projection layer 코드를 그대로 사용할 수 있음.
        위에 decode_token_ids_to_actions 함수와 같은 기능을 하지만,
        projection layer까지 거쳐서 최종적으로 qwen 모델의 입력 임베딩 차원인 2048로 변환된 action 임베딩을 반환하는 함수임.
        이를 qwen 모델의 텍스트/이미지 임베딩과 concat해서 모델 입력으로 사용할 예정임.
        """
        # 1. Token IDs → OpenVLA 임베딩
        bin_indices = self.OPENVLA_VOCAB_SIZE - action_token_ids  # 0~255

        print(f"action_token_ids: {action_token_ids}")
        print(f"bin_indices before clip: {bin_indices}")

        bin_indices = torch.clamp(bin_indices, min=0, max=255)
        
        openvla_embeds = self.openvla_embedding(
        bin_indices.clone().detach().to(dtype=torch.long).to("cuda")
        )  # (7, 4096)

        # 2. Projection Layer로 4096 → 2048 변환
        qwen_embeds = self.projection(openvla_embeds.float())  # (7, 2048)

        return qwen_embeds.unsqueeze(0).to(torch.bfloat16)    # (1, 7, 2048) ← 추가

    
    # 마지막 qwen 모델에서 생성된 action token ID 7개를 action vector 값 7개로 변환하는 함수.
    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        openvla에서 가져온 action detokenizer 코드임.
        qwen 모델에서 받은 action token은 openvla에서 생성된 token ID와 동일한 방식으로 binning되어 있기 때문에
        openvla의 detokenizer 코드를 그대로 사용할 수 있음.
        decode_token_ids_to_actions 함수는 action token ID 7개를 action vector 값 7개로 변환하는 역할을 함.

        여기서 OPENVLA_VOCAB_SIZE는 32000으로, action token ID는 31744~31999 사이의 값이 될 것임. 
        따라서 bin_indices는 0~255 사이의 값이 됨.
        openvla를 기준으로 하는 이유는 qwen 모델에서 생성된 action token ID가 openvla에서 생성된 token ID와 동일한 방식으로 binning되어 있고
        같은 임베딩 레이어를 공유하기 때문임. 
        따라서 action token ID → bin index → action value로 변환하는 과정은 openvla의 detokenizer 코드와 동일하게 적용할 수 있음.
        """
        QWEN_ACTION_TOKEN_START = len(self.processor.tokenizer) - 256  # 151665
        discretized_actions = action_token_ids - QWEN_ACTION_TOKEN_START  # 0~255
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]
    
    def concat_embeddings(self, text_image_embeds, action_embeds):
        """
        텍스트/이미지 임베딩 + action 임베딩 concat
        """
        return torch.cat([text_image_embeds, action_embeds], dim=1)
    
    # 아래 두 함수는 해당 클래스에서 사용되는 함수들을 논리 구조대로 묶어서 작성함.
    # 해당 클래스 사용시 두 함수를 중점적으로 사용하면 됨.
    def setup(self, openvla_embed_weights):
        """
        1. qwen 모델의 tokenizer vocab에 action token 256개 추가
        2. qwen 모델의 임베딩 레이어 크기를 tokenizer vocab 크기에 맞게 조정
        3. 새로 추가된 action token 256개를 OpenVLA 임베딩으로 초기화
        따로 함수로 빼는 이유는, setup 함수는 모델 초기화 시에 한 번만 호출되고,
        forward 함수는 매 추론마다 호출되기 때문에, 초기화 코드를 setup 함수로 분리해서 작성하는 것이 코드 구조상 더 깔끔하다고 생각했음.
        """
        self.add_tokenizer_vocab()
        self.resize_embeddings()
        self.init_action_embeddings(openvla_embed_weights)
        print("ActorActionTokenizer setup complete")

    def forward(self, action_token_ids, text_image_embeds):
        """
        action token ID 7개 → action 임베딩 → 텍스트/이미지 임베딩과 concat → qwen 모델 입력 임베딩
        얘는 매 추론마다 호출되는 함수로, action token ID 7개를 받아서 최종적으로 qwen 모델의 입력 임베딩으로 변환하는 역할을 함.
        그리고 텍스트/이미지 임베딩과 concat해서 모델 입력으로 사용하는 최종 함수.
        """
        action_embeds = self.embed_action_tokens(action_token_ids)  # (7, 2048)
        return self.concat_embeddings(text_image_embeds, action_embeds)
