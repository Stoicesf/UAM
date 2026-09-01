"""Actor 网络 — 输入 obs (+ Φ for DSGF)。"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """高斯策略 Actor。"""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Normal:
        h = self.net(x)
        return Normal(self.mean(h), self.log_std.exp())
