"""SECDO latent transition."""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentTransition(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))

    def predict_next_feat(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        z_next = self.forward(z, a)
        return self.decoder(z_next)
