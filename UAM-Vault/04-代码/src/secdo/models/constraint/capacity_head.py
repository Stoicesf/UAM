"""Capacity / constraint dynamics head: h_t [, c_t] → ĉ_{t+k}.

Frozen interface for Thm1 δ term. Residual default: ĉ = c + Δ(h).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CapacityHead(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        c_dim: int = 1,
        hidden: int = 64,
        horizon_k: int = 1,
        residual: bool = True,
        delta_scale: float = 0.5,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.horizon_k = horizon_k
        self.residual = residual
        self.delta_scale = delta_scale
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c_dim * horizon_k),
        )

    def forward(self, h: torch.Tensor, c_curr: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.net(h)
        if self.residual:
            if c_curr is None:
                raise ValueError("residual CapacityHead requires c_curr")
            c_base = c_curr.view(-1, self.c_dim)
            delta = self.delta_scale * torch.tanh(raw.view(-1, self.c_dim))
            out = (c_base + delta).clamp(min=1e-3)
        else:
            out = torch.nn.functional.softplus(raw)
            if self.horizon_k == 1:
                return out if out.dim() == 2 else out.view(-1, self.c_dim)
            return out.view(-1, self.horizon_k, self.c_dim)
        if self.horizon_k == 1:
            return out
        return out.view(-1, self.horizon_k, self.c_dim)


class ConstraintDynamics(nn.Module):
    """Paper-facing name: F_φ(state_history) → ĉ."""

    def __init__(self, head: CapacityHead):
        super().__init__()
        self.head = head

    def forward(
        self,
        state_history: torch.Tensor,
        c_curr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            state_history: latent h_t (or feature) [B, H]
            c_curr: current teacher capacity for residual head
        """
        return self.head(state_history, c_curr)


# Legacy alias (identical weights layout)
ConstraintHead = CapacityHead
