import numpy as np
import torch
import torch.nn as nn



class Projection(nn.Module):
    def __init__(self, openvla_dim = 4096, qwen_dim = 2048):
        super(Projection, self).__init__()
        """ 
        LayerNorm + MLP Projection으로 OpenVLA 임베딩 → Qwen 모델 입력 임베딩으로 변환
        projection layer에서 사용한 모델은 LLaVA에서 사용한 projection layer와 동일한 구조로
        LayerNorm + linear + GELU + linear로 구성 (LLaVA 논문 참고)
        """
        self.norm = nn.LayerNorm(openvla_dim) 

        self.projection = nn.Sequential(
            nn.Linear(openvla_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, qwen_dim)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.projection(x)
    