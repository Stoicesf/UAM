"""Enc → GRU → CapacityHead / state decoder (constraint dynamics stack)."""

from __future__ import annotations

import torch
import torch.nn as nn

from secdo.models.constraint.capacity_head import CapacityHead
from secdo.models.dynamics.encoder import Encoder
from secdo.models.dynamics.gru import GRUDynamics


class Predictor(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        latent_dim: int = 32,
        hidden: int = 64,
        residual: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(feat_dim, latent_dim=latent_dim, hidden=hidden)
        self.gru = GRUDynamics(latent_dim)
        # Name must stay constraint_head for legacy checkpoint keys
        self.constraint_head = CapacityHead(
            feat_dim=latent_dim, c_dim=1, hidden=hidden, residual=residual
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_dim),
        )

    def forward_step(
        self,
        feat: torch.Tensor,
        c_curr: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.encoder(feat)
        h = self.gru(z, h)
        c_hat = self.constraint_head(h, c_curr)
        s_hat = self.state_decoder(h)
        return {"z": z, "h": h, "c_hat": c_hat, "s_hat": s_hat}

    def predict_constraint(
        self,
        feat: torch.Tensor,
        c_curr: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward_step(feat, c_curr, h)
        return out["c_hat"], out["h"]


SecdoPredictor = Predictor
