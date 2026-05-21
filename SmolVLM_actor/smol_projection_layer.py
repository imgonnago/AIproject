import torch
import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, openvla_dim=4096, smol_dim=960):
        """
        OpenVLA 임베딩 → SmolVLM2 LLM 입력 임베딩으로 변환하는 Projection Layer.
        LLaVA 기반 구조: LayerNorm + Linear + GELU + Linear

        변경사항 (Qwen → SmolVLM2):
            qwen_dim: 2048 → smol_dim: 960
            SmolLM2-360M의 hidden_size = 960

        :param openvla_dim: OpenVLA LLM hidden size (LLaMA-2 기반, 4096)
        :param smol_dim: SmolVLM2 LLM hidden size (SmolLM2-360M 기반, 960)
        """
        super(Projection, self).__init__()

        self.norm = nn.LayerNorm(openvla_dim)

        self.projection = nn.Sequential(
            nn.Linear(openvla_dim, smol_dim),
            nn.GELU(),
            nn.Linear(smol_dim, smol_dim)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.projection(x)