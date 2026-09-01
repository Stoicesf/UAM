"""Learnable pairwise communication gate g_ij ∈ [0, 1].

AC-DSGF: Ã'_ij = A_ij · q_ij · g_ij
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CommunicationController(nn.Module):
    """Decide whether agent i should communicate with agent j.

    Expected edge_state last-dim features (v0):
        [distance, signal_quality, task_priority, resource]
    Missing channels may be zero-filled by the caller.
    """

    def __init__(self, hidden_dim: int = 128, edge_state_dim: int = 4):
        super().__init__()
        in_dim = 2 * hidden_dim + edge_state_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_state: torch.Tensor,
        adj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N, H) node embeddings
            edge_state: (B, N, N, E)
            adj_mask: optional (B, N, N) hard radius mask (1 = allowed)

        Returns:
            g: (B, N, N) communication gates in (0, 1), diagonal 0
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if edge_state.dim() == 3:
            edge_state = edge_state.unsqueeze(0)

        b, n, hid = h.shape
        h_i = h.unsqueeze(2).expand(b, n, n, hid)
        h_j = h.unsqueeze(1).expand(b, n, n, hid)
        x = torch.cat([h_i, h_j, edge_state], dim=-1)
        logits = self.encoder(x).squeeze(-1)
        g = torch.sigmoid(logits)

        eye = torch.eye(n, device=g.device, dtype=g.dtype).unsqueeze(0)
        g = g * (1.0 - eye)
        if adj_mask is not None:
            g = g * adj_mask.float()
        return g
