"""Dynamic Communication-aware Graph — quality-weighted adjacency."""

from __future__ import annotations

import torch
import torch.nn as nn


def _pairwise_dist(pos: torch.Tensor) -> torch.Tensor:
    diff = pos.unsqueeze(2) - pos.unsqueeze(1)
    return diff.norm(dim=-1)


def _velocity_angle(v: torch.Tensor) -> torch.Tensor:
    """Pairwise velocity alignment (B, N, N) mapped to [0, 1]."""
    v_i = v.unsqueeze(2)
    v_j = v.unsqueeze(1)
    cos = (v_i * v_j).sum(dim=-1)
    norm = v_i.norm(dim=-1) * v_j.norm(dim=-1)
    cos = cos / (norm + 1e-8)
    return (cos + 1.0) * 0.5


class DynamicGraphModule(nn.Module):
    """Build quality-weighted adjacency Ã_ij = A_ij · q_ij.

    Edge features: e_ij = [d_ij, Δv_ij, θ_ij, c_ij]
        d  — distance
        Δv — velocity difference magnitude
        θ  — velocity alignment (0~1)
        c  — prior link quality from distance decay
    """

    def __init__(self, comm_radius: float = 0.5, hidden_dim: int = 32, use_quality: bool = True):
        super().__init__()
        self.comm_radius = comm_radius
        self.use_quality = use_quality
        self.edge_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        comm_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            positions: (B, N, 2)
            velocities: (B, N, 2)
            comm_mask: optional (B, N, N) hard mask (e.g. occlusion), 1=allowed

        Returns:
            weighted_adj: (B, N, N) Ã_ij = A_ij · q_ij, diagonal zero
            edge_quality: (B, N, N) q_ij in (0, 1)
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
            velocities = velocities.unsqueeze(0)

        dist = _pairwise_dist(positions)
        delta_v = (velocities.unsqueeze(2) - velocities.unsqueeze(1)).norm(dim=-1)
        theta = _velocity_angle(velocities)
        c_prior = torch.exp(-(dist ** 2) / (2 * self.comm_radius ** 2 + 1e-8))

        edge_feat = torch.stack([dist, delta_v, theta, c_prior], dim=-1)
        if self.use_quality:
            q_ij = torch.sigmoid(self.edge_mlp(edge_feat).squeeze(-1))
        else:
            q_ij = torch.ones_like(dist)

        adj = (dist < self.comm_radius).float()
        eye = torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
        adj = adj * (1.0 - eye)

        if comm_mask is not None:
            adj = adj * comm_mask.float()

        weighted_adj = adj * q_ij
        return weighted_adj, q_ij
