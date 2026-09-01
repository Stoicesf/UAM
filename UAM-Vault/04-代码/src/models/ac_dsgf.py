"""AC-DSGF — Adaptive Communication-aware Dynamic Sparse Graph Fusion.

Wraps frozen DSGF spatial-temporal stack with a learnable communication gate.

    Ã' = A · q · g
    L  = L_ppo + λ_c · C_comm   (wired in guided runner, not here)

This module returns guidance field Φ plus communication diagnostics.
Full residual actor remains in guided MAPPO (unchanged DSGF path).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.communication.budget_layer import (
    apply_topk_budget,
    apply_topk_fixed_k,
    budget_edge_count,
    degree_budget_violation,
    hard_adjacency,
)
from models.communication.controller import CommunicationController
from models.communication.cost import communication_cost
from models.communication.scheduler import CommunicationScheduler
from models.dynamic_graph import DynamicGraphModule
from models.sparse_attention import CommunicationAwareSparseAttention
from models.temporal_encoder import TemporalEncoder


class ACDSGF(nn.Module):
    """Observation → (Φ, g, diagnostics)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        guidance_dim: int = 6,
        comm_radius: float = 0.5,
        num_heads: int = 4,
        num_spatial_layers: int = 2,
        use_temporal: bool = True,
        edge_state_dim: int = 4,
        lambda_c: float = 1e-3,
    ):
        super().__init__()
        self.comm_radius = comm_radius
        self.use_temporal = use_temporal
        self.guidance_dim = guidance_dim

        self.node_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
        )
        self.dynamic_graph = DynamicGraphModule(
            comm_radius=comm_radius, use_quality=True
        )
        self.controller = CommunicationController(
            hidden_dim=hidden_dim, edge_state_dim=edge_state_dim
        )
        self.scheduler = CommunicationScheduler(lambda_c=lambda_c)
        self.budget_ratio: float | None = None  # set at eval for budget sweep
        self.fixed_k: int | None = None  # T-RO Thm.~1: hard degree budget K
        # Eval ablations (Silence Collapse): None|"full"|"no_budget"|"random"
        self.ablation_mode: str | None = None
        self.spatial_layers = nn.ModuleList(
            [
                CommunicationAwareSparseAttention(hidden_dim, num_heads)
                for _ in range(num_spatial_layers)
            ]
        )
        self.temporal = TemporalEncoder(hidden_dim, hidden_dim) if use_temporal else None
        self.phi_head = nn.Linear(hidden_dim, guidance_dim)

    def _edge_state(
        self,
        positions: torch.Tensor,
        q_ij: torch.Tensor,
    ) -> torch.Tensor:
        """Build (B,N,N,4): distance, quality, priority_placeholder, resource_placeholder."""
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = diff.norm(dim=-1, keepdim=True)
        zeros = torch.zeros_like(dist)
        return torch.cat([dist, q_ij.unsqueeze(-1), zeros, zeros], dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
        z_prev: torch.Tensor | None = None,
        apply_budget: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if positions is None:
            positions = obs[..., :2]
        elif positions.dim() == 2:
            positions = positions.unsqueeze(0)

        velocities = obs[..., 2:4]
        weighted_adj, q_ij = self.dynamic_graph(positions, velocities)
        # Hard mask from quality graph support (nonzeros of radius mask approx)
        adj_mask = (weighted_adj > 0).float()
        # Restore binary radius if quality zeroed edges: use distance mask
        dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
        radius_mask = (dist < self.comm_radius).float()
        eye = torch.eye(radius_mask.shape[-1], device=obs.device).unsqueeze(0)
        radius_mask = radius_mask * (1.0 - eye)

        h = self.node_embed(obs)
        edge_state = self._edge_state(positions, q_ij)
        mode = (self.ablation_mode or "full").lower()
        if mode == "no_budget":
            # Open all radius edges → tests whether sparsity is necessary
            g = radius_mask
        elif mode == "random":
            # Random gates on radius support → not just fewer edges, but *learned* selection
            g = radius_mask * torch.rand_like(radius_mask)
        else:
            g = self.controller(h, edge_state, adj_mask=radius_mask)

        ratio = self.budget_ratio
        k_fixed = self.fixed_k
        if mode == "full" and (
            apply_budget
            or k_fixed is not None
            or (ratio is not None and ratio < 1.0)
        ):
            if k_fixed is not None:
                g = apply_topk_fixed_k(g, radius_mask, int(k_fixed))
            else:
                if ratio is None:
                    ratio = self.scheduler.budget_ratio
                if ratio is not None and ratio < 1.0:
                    g = apply_topk_budget(g, radius_mask, float(ratio))

        # Eval-only: multiply gates (e.g. leave-one-edge-out importance). No training use.
        mul = getattr(self, "eval_gate_multiplier", None)
        if mul is not None:
            m = mul.to(device=g.device, dtype=g.dtype)
            if m.dim() == 2:
                m = m.unsqueeze(0)
            g = g * m

        A_tilde = radius_mask * q_ij * g
        A_hard = hard_adjacency(g)

        x = h
        for layer in self.spatial_layers:
            x = torch.tanh(layer(x, A_tilde) + x)

        if self.temporal is not None:
            z = self.temporal(x, z_prev)
        else:
            z = x
        phi = self.phi_head(z)
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        feas = degree_budget_violation(
            A_hard,
            radius_mask,
            budget_ratio=None if k_fixed is not None else (
                float(ratio) if ratio is not None else 1.0
            ),
            k_fixed=int(k_fixed) if k_fixed is not None else None,
        )

        diagnostics = {
            "g": g,
            "A_tilde": A_tilde,
            "A_t": A_hard,
            "radius_mask": radius_mask,
            "S_t": g.detach(),  # post-controller / pre-or-post projection scores used
            "comm_cost": communication_cost(g),
            "hard_edges": torch.tensor(budget_edge_count(g), device=g.device),
            "q": q_ij,
            "budget_ratio": torch.tensor(
                float(ratio) if ratio is not None else 1.0, device=g.device
            ),
            "fixed_k": torch.tensor(
                float(k_fixed) if k_fixed is not None else -1.0, device=g.device
            ),
            "V_B": torch.tensor(feas["V_B"], device=g.device),
            "V_B_mean": torch.tensor(feas["V_B_mean"], device=g.device),
            "C_t": torch.tensor(feas["C"], device=g.device),
            "B_t": torch.tensor(feas["B_t"], device=g.device),
            "mean_degree": torch.tensor(feas["mean_degree"], device=g.device),
            "rho_t": torch.tensor(feas["rho"], device=g.device),
            "degrees": feas["degrees"],
            "k_caps": feas["k_caps"],
        }
        return phi, g, diagnostics

    # --- Twin / shared-state helpers (T-RO Gate 2; eval-only) ---

    def _normalize_batch(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if positions is None:
            positions = obs[..., :2]
        elif positions.dim() == 2:
            positions = positions.unsqueeze(0)
        return obs, positions

    def build_shared_context(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Shared quantities for twin branches on the same s_t (no env step)."""
        obs, positions = self._normalize_batch(obs, positions)
        velocities = obs[..., 2:4]
        _, q_ij = self.dynamic_graph(positions, velocities)
        dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
        radius_mask = (dist < self.comm_radius).float()
        eye = torch.eye(radius_mask.shape[-1], device=obs.device).unsqueeze(0)
        radius_mask = radius_mask * (1.0 - eye)
        h = self.node_embed(obs)
        edge_state = self._edge_state(positions, q_ij)
        scores = self.controller(h, edge_state, adj_mask=radius_mask)
        return {
            "obs": obs,
            "positions": positions,
            "h": h,
            "q_ij": q_ij,
            "radius_mask": radius_mask,
            "edge_state": edge_state,
            "scores": scores,
        }

    def decode_from_gate(
        self,
        ctx: dict[str, torch.Tensor],
        g: torch.Tensor,
        z_prev: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Aggregate messages under a fixed gate/adjacency on shared context."""
        h = ctx["h"]
        q_ij = ctx["q_ij"]
        radius_mask = ctx["radius_mask"]
        A_tilde = radius_mask * q_ij * g
        A_hard = hard_adjacency(g)
        x = h
        for layer in self.spatial_layers:
            x = torch.tanh(layer(x, A_tilde) + x)
        if self.temporal is not None:
            z = self.temporal(x, z_prev)
        else:
            z = x
        phi = self.phi_head(z)
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        return {
            "phi": phi,
            "message": x,  # post-attention embedding = M(G; s)
            "g": g,
            "A_t": A_hard,
            "A_tilde": A_tilde,
        }

    def twin_forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        sparse_mode: str = "fixed_k",
        k_fixed: int | None = None,
        budget_ratio: float | None = None,
        z_prev: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Shared-state twin: G* = full-support reference BEFORE budget projection.

        G*_t uses pre-projection scores on the geometric support (not unrestricted
        all-to-all / infinite communication). Sparse G_t = Π_B(S_t).

        sparse_mode:
          - "fixed_k": Top-K on learned scores (Gate 1)
          - "ratio": apply_topk_budget
          - "empty": no edges
          - "full": sparse == reference (sanity → ε_G ≈ 0)
        """
        ctx = self.build_shared_context(obs, positions)
        scores = ctx["scores"]
        mask = ctx["radius_mask"]
        # Full-support reference topology before budget projection.
        g_full = scores * mask

        if sparse_mode == "full":
            g_sparse = g_full
        elif sparse_mode == "empty":
            g_sparse = torch.zeros_like(g_full)
        elif sparse_mode == "ratio":
            ratio = budget_ratio if budget_ratio is not None else self.budget_ratio
            if ratio is None or ratio >= 1.0:
                g_sparse = g_full
            else:
                g_sparse = apply_topk_budget(scores, mask, float(ratio))
        else:  # fixed_k
            K = k_fixed if k_fixed is not None else self.fixed_k
            if K is None:
                raise ValueError("twin_forward fixed_k requires k_fixed or encoder.fixed_k")
            g_sparse = apply_topk_fixed_k(scores, mask, int(K))

        full = self.decode_from_gate(ctx, g_full, z_prev=z_prev)
        sparse = self.decode_from_gate(ctx, g_sparse, z_prev=z_prev)
        return {
            "ctx": ctx,
            "full": full,
            "sparse": sparse,
            "G_star": hard_adjacency(g_full),
            "G_t": hard_adjacency(g_sparse),
        }
