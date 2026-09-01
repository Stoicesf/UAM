"""引导场融合 — 将 Φ 与原始观测拼接供 Actor 使用。

Actor 输入: obs (64维) + Φ (4维) = 68维
MAPPO 框架本身无需修改。
"""

from __future__ import annotations

import torch


def fuse_obs_guidance(
    obs: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """拼接观测与引导向量。

    Args:
        obs: (B, N, obs_dim) 或 (B*N, obs_dim)
        phi: (B, N, guidance_dim) 或 (B*N, guidance_dim)

    Returns:
        fused: 拼接后的 Actor 输入
    """
    return torch.cat([obs, phi], dim=-1)
