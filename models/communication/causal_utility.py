"""Action-Causal Utility — predict whether message i→j changes j's action.

AC-DSGF++: U_ij ∈ (0,1) ≈ P(‖a_j^m − a_j^0‖ > 0 | h_i, h_j)

v1 CommunicationController is untouched; this module is ++ only.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CausalUtility(nn.Module):
    """Pairwise utility head: (h_i, h_j) → U_ij ∈ (0, 1)."""

    def __init__(self, hidden_dim: int = 128, mid_dim: int = 128):
        super().__init__()
        self.utility_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, N, H) or (N, H) node embeddings

        Returns:
            U: (B, N, N) pairwise utilities; diagonal forced to 0
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)
        b, n, hid = h.shape
        h_i = h.unsqueeze(2).expand(b, n, n, hid)
        h_j = h.unsqueeze(1).expand(b, n, n, hid)
        x = torch.cat([h_i, h_j], dim=-1)
        u = self.utility_net(x).squeeze(-1)
        eye = torch.eye(n, device=u.device, dtype=u.dtype).unsqueeze(0)
        return u * (1.0 - eye)

    def forward_pair(
        self, h_i: torch.Tensor, h_j: torch.Tensor
    ) -> torch.Tensor:
        """Single-pair API: h_i, h_j last-dim H → U scalar (or batched)."""
        x = torch.cat([h_i, h_j], dim=-1)
        return self.utility_net(x).squeeze(-1)
