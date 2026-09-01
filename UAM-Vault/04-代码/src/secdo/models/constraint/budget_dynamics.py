"""Learned constraint-parameter dynamics F_φ."""

from __future__ import annotations

import torch
import torch.nn as nn


class BudgetDynamics(nn.Module):
    """Predict next scalar (or vector) constraint parameter c."""

    def __init__(self, feat_dim: int, c_dim: int = 1, hidden: int = 64):
        super().__init__()
        self.c_dim = c_dim
        self.net = nn.Sequential(
            nn.Linear(c_dim + feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c_dim),
            nn.Softplus(),
        )

    def forward(self, c_t: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        if c_t.dim() == 1:
            c_t = c_t.unsqueeze(-1)
        return self.net(torch.cat([c_t, s_feat], dim=-1))
