"""Stage 2 — GAT Guide over dynamic communication graph.

Each agent i:
    x_i = obs_i  (position, velocity, goal, lidar, ...)
    A_ij = 1  if ||p_i - p_j|| < R_c

    h_i = sum_j alpha_ij W h_j
    Phi_i = (dx, dy, risk, priority)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from guidance.graph_builder import build_adjacency


class GATLayer(nn.Module):
    """Single-head graph attention with adjacency mask."""

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim))
        self.a_dst = nn.Parameter(torch.empty(out_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            adj: (B, N, N) neighbor mask (0/1), no self-loops
        """
        h = self.W(x)
        e = self.leaky_relu(
            self.a_src.view(1, 1, 1, -1) * h.unsqueeze(2)
            + self.a_dst.view(1, 1, 1, -1) * h.unsqueeze(1)
        ).sum(dim=-1)

        mask = adj <= 0
        e = e.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        return torch.matmul(alpha, h)


class GATGuideEncoder(nn.Module):
    """Dynamic-graph GAT guide field encoder — Stage 2."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        guidance_dim: int = 4,
        comm_radius: float = 0.5,
        num_gat_layers: int = 1,
    ):
        super().__init__()
        self.comm_radius = comm_radius
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.gat_layers = nn.ModuleList(
            [GATLayer(hidden_dim, hidden_dim) for _ in range(num_gat_layers)]
        )
        self.guidance_head = nn.Linear(hidden_dim, guidance_dim)
        self.budget_ratio: float | None = None

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            obs: (B, N, obs_dim) or (N, obs_dim)
            positions: (B, N, 2) — defaults to obs[..., :2]
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if positions is None:
            positions = obs[..., :2]
        elif positions.dim() == 2:
            positions = positions.unsqueeze(0)

        adj = build_adjacency(positions, self.comm_radius)
        if self.budget_ratio is not None and self.budget_ratio < 1.0:
            from models.communication.budget_layer import apply_topk_budget

            dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
            # Prefer closer neighbors under hard budget
            scores = adj * (1.0 / (dist + 1e-6))
            adj = (apply_topk_budget(scores, adj, float(self.budget_ratio)) > 0).float()

        x = torch.tanh(self.input_proj(obs))
        for layer in self.gat_layers:
            x = torch.tanh(layer(x, adj) + x)

        phi = self.guidance_head(x)
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        return phi
