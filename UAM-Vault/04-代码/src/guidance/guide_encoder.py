"""Stage 1 — Guide Field MLP: obs -> Phi [dx, dy, risk, priority].

No Transformer, no Attention, no Dynamic Graph.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GuideEncoder(nn.Module):
    """Per-agent MLP guide encoder — Stage 1 only."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        guidance_dim: int = 4,
    ):
        super().__init__()
        self.guidance_dim = guidance_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, guidance_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (..., obs_dim)

        Returns:
            phi: (..., guidance_dim)
        """
        phi = self.net(obs)
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        return phi
