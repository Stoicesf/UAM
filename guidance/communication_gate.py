"""通信门控 — 控制通信开销统计（实验3: Communication Cost）。

用于记录/限制每步实际通信边数，支持论文通信效率分析。
"""

from __future__ import annotations

import torch


def count_communication_cost(adj: torch.Tensor) -> torch.Tensor:
    """统计每 batch 的通信边数（无向图，除以 2）。

    Args:
        adj: (B, N, N) 邻接矩阵

    Returns:
        cost: (B,) 每环境的通信边数
    """
    return adj.sum(dim=(1, 2)) / 2.0


def communication_ratio(adj: torch.Tensor) -> float:
    """通信边数占完全图比例，用于指标汇报。"""
    b, n, _ = adj.shape
    max_edges = n * (n - 1) / 2
    return (adj.sum().item() / 2) / (b * max_edges) if max_edges > 0 else 0.0
