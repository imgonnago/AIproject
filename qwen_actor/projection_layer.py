import numpt as np
import torch
import torch.nn as nn



class Projection(nn.Module):
    def __init__(self, openvla_dim, qwen_dim):
        super(Projection, self).__init__()
        openvla_dim = 4096
        qwen_dim = 2048

        self.norm = nn.LayerNorm(openvla_dim)

        self.projection = nn.Sequential(
            nn.Linear(openvla_dim, qwen_dim),
            nn.GELU(),
            nn.Linear(qwen_dim, qwen_dim)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.projection(x)
    