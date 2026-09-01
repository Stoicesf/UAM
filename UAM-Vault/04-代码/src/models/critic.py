"""Critic 网络 — 集中式价值函数 (MAPPO)。"""

from __future__ import annotations

import torch
import torch.nn as nn


class Critic(nn.Module):
    """全局状态价值估计。"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
