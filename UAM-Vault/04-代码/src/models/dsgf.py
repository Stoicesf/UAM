"""DSGF — Dynamic Spatial-Temporal Graph Fusion guidance field."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.dynamic_graph import DynamicGraphModule
from models.sparse_attention import CommunicationAwareSparseAttention
from models.temporal_encoder import TemporalEncoder


class DSGF(nn.Module):
    """Dynamic Sparse Graph Fusion encoder: obs → Φ_i.

    Pipeline:
        agent state → Dynamic Graph → Sparse Spatial Attention
                    → Temporal Memory → Φ = [dir(2), risk, priority, density, comm_q]
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        guidance_dim: int = 6,
        comm_radius: float = 0.5,
        num_heads: int = 4,
        num_spatial_layers: int = 2,
        use_dynamic_quality: bool = True,
        use_temporal: bool = True,
    ):
        super().__init__()
        self.guidance_dim = guidance_dim
        self.comm_radius = comm_radius
        self.use_temporal = use_temporal

        self.node_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
        )
        self.dynamic_graph = DynamicGraphModule(
            comm_radius=comm_radius, use_quality=use_dynamic_quality
        )
        self.spatial_layers = nn.ModuleList([
            CommunicationAwareSparseAttention(hidden_dim, num_heads)
            for _ in range(num_spatial_layers)
        ])
        if use_temporal:
            self.temporal = TemporalEncoder(hidden_dim, hidden_dim)
        else:
            self.temporal = None
        self.phi_head = nn.Linear(hidden_dim, guidance_dim)
        # Optional eval-time hard budget (None = frozen DSGF behavior unchanged)
        self.budget_ratio: float | None = None

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
        comm_mask: torch.Tensor | None = None,
        z_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: (B, N, obs_dim) or (N, obs_dim)
            positions: optional (B, N, 2), default obs[..., :2]
            comm_mask: optional (B, N, N) communication allowance
            z_prev: optional temporal hidden state

        Returns:
            phi: (B, N, guidance_dim)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        if positions is None:
            positions = obs[..., :2]
        velocities = obs[..., 2:4]

        weighted_adj, _ = self.dynamic_graph(positions, velocities, comm_mask)

        if self.budget_ratio is not None and self.budget_ratio < 1.0:
            from models.communication.budget_layer import apply_topk_budget

            mask = (weighted_adj > 0).float()
            weighted_adj = apply_topk_budget(weighted_adj, mask, float(self.budget_ratio))

        x = self.node_embed(obs)
        for layer in self.spatial_layers:
            x = torch.tanh(layer(x, weighted_adj) + x)

        if self.temporal is not None:
            z = self.temporal(x, z_prev)
        else:
            z = x
        phi = self.phi_head(z)

        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        return phi
