"""引导对齐奖励 — R_guide = cos(θ_action - θ_Φ)。

对应论文:
    R = R_goal + R_collision + R_guide
"""

from __future__ import annotations

import torch


def guidance_alignment_reward(
    actions: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """计算动作方向与引导方向的对齐度。

    Args:
        actions: (..., 2) 动作向量 [vx, vy] 或类似
        phi: (..., 4) 引导向量 [dx, dy, risk, priority]

    Returns:
        reward: (...) 对齐奖励 ∈ [-1, 1]
    """
    action_dir = actions[..., :2]
    guide_dir = phi[..., :2]

    action_norm = action_dir / (action_dir.norm(dim=-1, keepdim=True) + 1e-8)
    guide_norm = guide_dir / (guide_dir.norm(dim=-1, keepdim=True) + 1e-8)

    return (action_norm * guide_norm).sum(dim=-1)
