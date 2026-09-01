"""Causal-aware communication gate for AC-DSGF++ only.

g_ij = σ(W[h_i, h_j, edge_state, U_ij])
Optional residual mix with a base (non-utility) gate to avoid cold-start collapse.

Does NOT replace models/communication/controller.py (v1 frozen).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalCommunicationController(nn.Module):
    """Gate conditioned on predicted action-causal utility."""

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_state_dim: int = 4,
        residual: bool = True,
        residual_alpha: float = 0.5,
    ):
        super().__init__()
        self.residual = residual
        self.residual_alpha = residual_alpha
        # edge_state + scalar utility
        in_dim = 2 * hidden_dim + edge_state_dim + 1
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Base gate without U — keeps early communication open
        base_in = 2 * hidden_dim + edge_state_dim
        self.base_encoder = nn.Sequential(
            nn.Linear(base_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_state: torch.Tensor,
        utility: torch.Tensor,
        adj_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N, H)
            edge_state: (B, N, N, E)
            utility: (B, N, N) U_ij ∈ [0, 1] — should be stop-grad from caller
            adj_mask: optional (B, N, N)

        Returns:
            g: (B, N, N) gates in (0, 1), diagonal 0
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
        if edge_state.dim() == 3:
            edge_state = edge_state.unsqueeze(0)
        if utility.dim() == 2:
            utility = utility.unsqueeze(0)

        b, n, hid = h.shape
        h_i = h.unsqueeze(2).expand(b, n, n, hid)
        h_j = h.unsqueeze(1).expand(b, n, n, hid)
        u = utility.unsqueeze(-1)
        x = torch.cat([h_i, h_j, edge_state, u], dim=-1)
        g = torch.sigmoid(self.encoder(x).squeeze(-1))

        if self.residual:
            x_base = torch.cat([h_i, h_j, edge_state], dim=-1)
            g_base = torch.sigmoid(self.base_encoder(x_base).squeeze(-1))
            a = float(self.residual_alpha)
            g = a * g + (1.0 - a) * g_base

        eye = torch.eye(n, device=g.device, dtype=g.dtype).unsqueeze(0)
        g = g * (1.0 - eye)
        if adj_mask is not None:
            g = g * adj_mask.float()
        return g

    def set_residual_alpha(self, alpha: float) -> None:
        self.residual_alpha = float(max(0.0, min(1.0, alpha)))
