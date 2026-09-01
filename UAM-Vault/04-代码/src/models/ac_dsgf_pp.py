"""AC-DSGF++ — Action-Causal Adaptive Communication DSGF.

Incremental upgrade over frozen AC-DSGF v1:
  1) CausalUtility predicts U_ij (will message change j's action?)
  2) Causal gate: g_ij = σ(W[h_i,h_j,d,ρ,U])
  3) Same budget + residual DSGF backbone as v1

v1 module `models/ac_dsgf.py` is never imported for mutation — logic is
duplicated/extended here so checkpoints and APIs stay separate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.communication.budget_layer import apply_topk_budget, budget_edge_count
from models.communication.causal_utility import CausalUtility
from models.communication.controller import CommunicationController
from models.communication.controller_causal import CausalCommunicationController
from models.communication.cost import communication_cost
from models.communication.loss_rank import utility_ranking_loss
from models.communication.scheduler import CommunicationScheduler
from models.dynamic_graph import DynamicGraphModule
from models.sparse_attention import CommunicationAwareSparseAttention
from models.temporal_encoder import TemporalEncoder
from utils.counterfactual import (
    broadcast_receiver_utility,
    pearson_correlation,
    phi_as_action_proxy,
)
from models.communication.causal_utility_v2 import (
    compute_cau_agent_utility,
    compute_cau_v2_agent_utility,
)


class ACDSGFpp(nn.Module):
    """Observation → (Φ, g, diagnostics) with action-causal utility."""

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
        gate_residual: bool = True,
        gate_residual_alpha: float = 0.5,
        use_utility_in_gate: bool = True,
        use_utility_in_policy: bool = True,
        utility_target: str = "action",
        cau_alpha: float = 0.5,
        cau_beta: float = 0.4,
        cau_gamma: float = 0.1,
        cau_horizon: int = 10,
        cau_td_gamma: float = 0.95,
    ):
        super().__init__()
        self.comm_radius = comm_radius
        self.use_temporal = use_temporal
        self.guidance_dim = guidance_dim
        self.hidden_dim = hidden_dim
        # Ablation: False → U = f(h.detach()) so L_u cannot reshape backbone/policy
        self.use_utility_in_gate = use_utility_in_gate
        self.use_utility_in_policy = use_utility_in_policy
        self.utility_target = str(utility_target).lower()
        self.cau_alpha = float(cau_alpha)
        self.cau_beta = float(cau_beta)
        self.cau_gamma = float(cau_gamma)
        self.cau_horizon = int(cau_horizon)
        self.cau_td_gamma = float(cau_td_gamma)

        self.node_embed = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
        )
        self.dynamic_graph = DynamicGraphModule(
            comm_radius=comm_radius, use_quality=True
        )
        self.causal_utility = CausalUtility(hidden_dim=hidden_dim)
        self.controller = CausalCommunicationController(
            hidden_dim=hidden_dim,
            edge_state_dim=edge_state_dim,
            residual=gate_residual,
            residual_alpha=gate_residual_alpha,
        )
        # Ablation: gate without utility signal (v1-style)
        self.controller_v1 = CommunicationController(
            hidden_dim=hidden_dim, edge_state_dim=edge_state_dim
        )
        self.scheduler = CommunicationScheduler(lambda_c=lambda_c)
        self.budget_ratio: float | None = None
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
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dist = diff.norm(dim=-1, keepdim=True)
        zeros = torch.zeros_like(dist)
        return torch.cat([dist, q_ij.unsqueeze(-1), zeros, zeros], dim=-1)

    def _radius_mask(self, positions: torch.Tensor) -> torch.Tensor:
        dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
        radius_mask = (dist < self.comm_radius).float()
        eye = torch.eye(radius_mask.shape[-1], device=positions.device).unsqueeze(0)
        return radius_mask * (1.0 - eye)

    def _aggregate(
        self,
        h: torch.Tensor,
        A_tilde: torch.Tensor,
        z_prev: torch.Tensor | None,
    ) -> torch.Tensor:
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
        return phi

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
        z_prev: torch.Tensor | None = None,
        apply_budget: bool = False,
        force_zero_comm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if positions is None:
            positions = obs[..., :2]
        elif positions.dim() == 2:
            positions = positions.unsqueeze(0)

        velocities = obs[..., 2:4]
        weighted_adj, q_ij = self.dynamic_graph(positions, velocities)
        del weighted_adj  # support via radius_mask
        radius_mask = self._radius_mask(positions)

        h = self.node_embed(obs)
        edge_state = self._edge_state(positions, q_ij)
        mode = (self.ablation_mode or "full").lower()
        h_for_u = h if self.use_utility_in_policy else h.detach()
        u = self._utility_field(h_for_u, edge_state, radius_mask, mode)

        if force_zero_comm or mode == "zero":
            g = torch.zeros_like(radius_mask)
        elif mode == "no_budget":
            g = radius_mask
        elif mode == "random":
            g = radius_mask * torch.rand_like(radius_mask)
        elif mode == "distance":
            dist = edge_state[..., 0]
            g = radius_mask * torch.sigmoid(-5.0 * (dist - 0.25))
        elif (
            mode in ("wo_utility", "no_utility")
            or not self.use_utility_in_gate
        ):
            g = self.controller_v1(h, edge_state, adj_mask=radius_mask)
        else:
            # Stop-grad U → gate: utility learns from L_u / L_rank only
            g = self.controller(h, edge_state, u.detach(), adj_mask=radius_mask)

        ratio = self.budget_ratio
        if (
            not force_zero_comm
            and mode == "full"
            and (apply_budget or (ratio is not None and ratio < 1.0))
        ):
            if ratio is None:
                ratio = self.scheduler.budget_ratio
            if ratio is not None and ratio < 1.0:
                g = apply_topk_budget(g, radius_mask, float(ratio))

        A_tilde = radius_mask * q_ij * g
        phi = self._aggregate(h, A_tilde, z_prev)

        diagnostics = {
            "g": g,
            "U": u,
            "A_tilde": A_tilde,
            "comm_cost": communication_cost(g),
            "hard_edges": torch.tensor(budget_edge_count(g), device=g.device),
            "q": q_ij,
            "h": h,
            "budget_ratio": torch.tensor(
                float(ratio) if ratio is not None else 1.0, device=g.device
            ),
        }
        return phi, g, diagnostics

    def _utility_field(
        self,
        h: torch.Tensor,
        edge_state: torch.Tensor,
        radius_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == "random_utility":
            return radius_mask * torch.rand_like(radius_mask)
        if mode == "distance_utility":
            dist = edge_state[..., 0]
            return radius_mask * torch.sigmoid(-5.0 * (dist - 0.25))
        if mode in ("wo_utility", "no_utility"):
            return torch.zeros_like(radius_mask)
        return self.causal_utility(h)

    def compute_utility_targets(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
        residual_delta_fn=None,
        beta: float = 1.0,
        value_fn=None,
        action_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (U_pred, U_star.detach(), pair_mask).

        U* is always stop-grad so policy cannot cheat utility loss.

        utility_target:
          - 'action' (legacy): ‖a^m − a^0‖
          - 'cau': 0.5ΔR + 0.4ΔV + 0.1ΔA  (task-aware)
        """
        if positions is None:
            positions = obs[..., :2]

        # U_pred with grad (for utility head)
        phi_m, _, diag = self.forward(obs, positions, force_zero_comm=False)
        u_pred = diag["U"]

        # Counterfactual targets — fully detached (no fake causality)
        with torch.no_grad():
            phi_0, _, _ = self.forward(obs, positions, force_zero_comm=True)
            phi_m_det = phi_m.detach()

            if action_fn is not None:
                a_m = action_fn(obs, phi_m_det)
                a_0 = action_fn(obs, phi_0)
            elif residual_delta_fn is not None:
                # Do NOT multiply by β: schedule decay would kill U*
                a_m = residual_delta_fn(phi_m_det)
                a_0 = residual_delta_fn(phi_0)
            else:
                a_m, a_0 = phi_m_det, phi_0

            if self.utility_target in ("cau_v2", "v2", "outcome_v2"):
                # CAU-v2: signed+batch-norm + H-step ΔV
                # Weights: β·ΔV + α·ΔR + γ·ΔA  (default 0.6 / 0.35 / 0.05)
                u_agent = compute_cau_v2_agent_utility(
                    obs,
                    a_m,
                    a_0,
                    value_fn=value_fn,
                    alpha_v=self.cau_beta,
                    alpha_r=self.cau_alpha,
                    alpha_a=self.cau_gamma,
                    horizon=self.cau_horizon,
                    gamma=self.cau_td_gamma,
                )
            elif self.utility_target in ("cau", "outcome", "task"):
                u_agent = compute_cau_agent_utility(
                    obs,
                    a_m,
                    a_0,
                    value_fn=value_fn,
                    alpha=self.cau_alpha,
                    beta=self.cau_beta,
                    gamma=self.cau_gamma,
                )
            elif residual_delta_fn is not None or action_fn is not None:
                a_diff = a_m - a_0
                u_agent = torch.clamp(a_diff.norm(dim=-1), 0.0, 1.0)
            else:
                u_agent = phi_as_action_proxy(phi_0, phi_m_det)

            u_star = broadcast_receiver_utility(u_agent)

        if u_pred.shape != u_star.shape:
            u_star = u_star.expand_as(u_pred)

        radius_mask = self._radius_mask(positions)
        if radius_mask.dim() == 2:
            radius_mask = radius_mask.unsqueeze(0)
        eye = torch.eye(radius_mask.shape[-1], device=radius_mask.device).unsqueeze(0)
        pair_mask = radius_mask * (1.0 - eye)
        return u_pred, u_star.detach(), pair_mask

    def utility_diagnostics(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
        residual_delta_fn=None,
        beta: float = 1.0,
        u_thr: float = 0.3,
        g_thr: float = 0.5,
        value_fn=None,
        action_fn=None,
    ) -> dict[str, float]:
        u_pred, u_star, pair_mask = self.compute_utility_targets(
            obs,
            positions,
            residual_delta_fn=residual_delta_fn,
            beta=beta,
            value_fn=value_fn,
            action_fn=action_fn,
        )
        _, g, _ = self.forward(obs, positions)
        m = pair_mask > 0
        masked_u = u_pred[m]
        # Communication Utility Density (CUD) = Σ g·U* / Σ g
        g_safe = g.clamp_min(0.0)
        cud = float(
            (g_safe * u_star * pair_mask).sum()
            / (g_safe * pair_mask).sum().clamp_min(1e-8)
        )
        # Legacy hard precision (often ~0 under soft gates g≪0.5)
        sent = m & (g > g_thr)
        valuable = u_star > u_thr
        tp = (sent & valuable).sum().float()
        fp = (sent & ~valuable).sum().float()
        precision = float(tp / (tp + fp + 1e-8))
        # CEI proxy on this batch (caller may overwrite with episode stats)
        comm = float(g.sum(dim=(-2, -1)).mean())
        return {
            "utility_mean": float(masked_u.mean()) if masked_u.numel() else 0.0,
            "utility_std": float(masked_u.std(unbiased=False)) if masked_u.numel() > 1 else 0.0,
            "gate_mass": comm,
            "utility_corr": pearson_correlation(u_pred, u_star, pair_mask),
            "comm_precision": precision,
            "cud": cud,
        }

    def utility_loss(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor | None = None,
        residual_delta_fn=None,
        beta: float = 1.0,
        lambda_rank: float = 0.0,
        ranking_mode: str = "hinge",
        value_fn=None,
        action_fn=None,
    ) -> torch.Tensor:
        """λ_u path: MSE(U, U*) + optional ranking. U* always detached."""
        if positions is None:
            positions = obs[..., :2] if obs.dim() >= 2 else None

        u_pred, u_star, pair_mask = self.compute_utility_targets(
            obs,
            positions,
            residual_delta_fn=residual_delta_fn,
            beta=beta,
            value_fn=value_fn,
            action_fn=action_fn,
        )
        mse = F.mse_loss(u_pred, u_star.detach())
        if lambda_rank > 0:
            rank = utility_ranking_loss(
                u_pred, u_star, pair_mask, mode=ranking_mode
            )
            return mse + lambda_rank * rank
        return mse