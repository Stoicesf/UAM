"""h_t = GRU(z_t, h_{t-1})."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUDynamics(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.cell = nn.GRUCell(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(z.shape[0], self.latent_dim, device=z.device, dtype=z.dtype)
        return self.cell(z, h)


TinyRecurrent = GRUDynamics
