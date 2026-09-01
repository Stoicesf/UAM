"""观测处理 — 将 VMAS 原始 obs 统一为固定维度 (64维)。

Baseline 与 DSGF 共用同一套观测编码。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """VMAS raw obs → 固定维度特征。"""

    def __init__(self, raw_dim: int = 18, obs_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(raw_dim, obs_dim)

    def forward(self, raw_obs: torch.Tensor) -> torch.Tensor:
        return self.proj(raw_obs)


def stack_agent_obs(obs_list: list[torch.Tensor]) -> torch.Tensor:
    """将 VMAS 返回的 agent obs 列表堆叠为 (B, N, D)。"""
    return torch.stack(obs_list, dim=1)
