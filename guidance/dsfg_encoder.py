"""DSGF 编码器 — Swarm State → Compressed Guidance。

输出每架无人机的引导向量 Φ_i = [dx, dy, risk, priority]。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from guidance.graph_builder import build_adjacency
from guidance.sparse_attention import SparseAttention


class DSGFEncoder(nn.Module):
    """动态蜂群引导场编码器 — 论文核心模块。"""

    def __init__(
        self,
        node_dim: int = 64,
        hidden_dim: int = 128,
        guidance_dim: int = 4,
        comm_radius: float = 0.5,
        num_heads: int = 4,
    ):
        super().__init__()
        self.comm_radius = comm_radius
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.sparse_attn = SparseAttention(hidden_dim, num_heads)
        self.guidance_head = nn.Linear(hidden_dim, guidance_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, node_dim) 每架无人机的状态特征
            positions: (B, N, 2) 平面位置

        Returns:
            phi: (B, N, guidance_dim) 引导向量 Φ
        """
        adj = build_adjacency(positions, self.comm_radius)
        x = self.node_proj(node_features)
        x = self.sparse_attn(x, adj)
        phi = self.guidance_head(x)
        # 方向分量归一化到单位向量
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        return phi
