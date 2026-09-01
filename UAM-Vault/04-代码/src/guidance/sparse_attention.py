"""稀疏注意力 — 仅关注通信半径内的邻居节点。

对应论文: Sparse Attention over dynamic graph。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SparseAttention(nn.Module):
    """基于邻接矩阵 mask 的多头稀疏注意力。"""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) 节点特征
            adj: (B, N, N) 邻接矩阵，0 表示不可通信

        Returns:
            out: (B, N, D) 注意力输出
        """
        b, n, _ = x.shape
        # MultiheadAttention(batch_first) 需要 (B*num_heads, N, N)
        mask = adj == 0
        mask = (
            mask.unsqueeze(1)
            .expand(b, self.attn.num_heads, n, n)
            .reshape(b * self.attn.num_heads, n, n)
        )
        return self.attn(x, x, x, attn_mask=mask)[0]
