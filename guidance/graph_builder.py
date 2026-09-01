"""动态通信图构建 — 基于距离阈值的邻接矩阵。

每一步根据无人机位置计算:
    A_ij = 1  if dist(i, j) < R_c  else 0

对应论文: 动态稀疏通信拓扑。
"""

from __future__ import annotations

import torch


def build_adjacency(
    positions: torch.Tensor,
    comm_radius: float,
) -> torch.Tensor:
    """构建 batch 邻接矩阵。

    Args:
        positions: (B, N, 2) 无人机平面坐标
        comm_radius: 通信半径 R_c

    Returns:
        adj: (B, N, N) 布尔/浮点邻接矩阵，对角线为 0
    """
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 2)
    dist = torch.norm(diff, dim=-1)
    adj = (dist < comm_radius).float()
    eye = torch.eye(positions.shape[1], device=positions.device).unsqueeze(0)
    adj = adj * (1.0 - eye)
    return adj
