"""z_t = Enc(s_t) — MLP latent encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# Checkpoint / legacy aliases
LatentEncoder = Encoder
