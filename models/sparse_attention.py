"""Communication-aware sparse spatial attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommunicationAwareSparseAttention(nn.Module):
    """α_ij = softmax_j(Q_i K_j^T / √d + log q_ij), masked to neighbors only."""

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        weighted_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) node embeddings
            weighted_adj: (B, N, N) Ã_ij = A_ij · q_ij
        """
        b, n, _ = x.shape
        h = self.num_heads
        d = self.head_dim

        q = self.q_proj(x).view(b, n, h, d).transpose(1, 2)
        k = self.k_proj(x).view(b, n, h, d).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        log_q = torch.log(weighted_adj.clamp(min=1e-8)).unsqueeze(1)
        scores = scores + log_q

        no_edge = weighted_adj <= 0
        scores = scores.masked_fill(no_edge.unsqueeze(1), float("-inf"))
        alpha = F.softmax(scores, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        out = torch.matmul(alpha, v)
        out = out.transpose(1, 2).contiguous().view(b, n, self.embed_dim)
        return self.out_proj(out)
