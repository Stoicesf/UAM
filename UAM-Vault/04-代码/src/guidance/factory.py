"""Guidance encoder factory — select module by experiment stage."""

from __future__ import annotations

import torch
import torch.nn as nn

from guidance.dsfg_encoder import DSGFEncoder as LegacyDSGFEncoder
from guidance.guide_encoder import GuideEncoder
from guidance.sparse_attention import SparseAttention
from models.ac_dsgf import ACDSGF
from models.ac_dsgf_pp import ACDSGFpp
from models.dsgf import DSGF
from models.guide_gnn import GATGuideEncoder


class SparseGuideEncoder(nn.Module):
    """Stage 2 — self-attention over all agents (no dynamic graph yet)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        guidance_dim: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.proj = nn.Linear(obs_dim, hidden_dim)
        self.attn = SparseAttention(hidden_dim, num_heads)
        self.head = nn.Linear(hidden_dim, guidance_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Stage 2: fully connected attention mask (all agents visible)
        x = self.proj(obs)
        b, n, _ = x.shape
        adj = x.new_ones(b, n, n)
        eye = torch.eye(n, device=x.device).unsqueeze(0)
        adj = adj * (1.0 - eye)
        x = self.attn(x, adj)
        phi = self.head(x)
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        return phi


def build_guidance_encoder(cfg: dict, obs_dim: int) -> nn.Module | None:
    """Build guidance encoder from experiment config."""
    mode = cfg.get("mode", "none")
    hidden = cfg.get("hidden_dim", 128)
    guidance_dim = cfg.get("guidance_dim", 4)
    num_heads = cfg.get("num_heads", 4)
    comm_radius = cfg.get("comm_radius", 0.5)

    if mode == "none":
        return None
    if mode == "mlp":
        return GuideEncoder(obs_dim, hidden, guidance_dim)
    if mode == "gat":
        return GATGuideEncoder(
            obs_dim=obs_dim,
            hidden_dim=hidden,
            guidance_dim=guidance_dim,
            comm_radius=comm_radius,
            num_gat_layers=cfg.get("num_gat_layers", 1),
        )
    if mode == "sparse":
        return SparseGuideEncoder(obs_dim, hidden, guidance_dim, num_heads)
    if mode == "graph":
        return LegacyDSGFEncoder(
            node_dim=obs_dim,
            hidden_dim=hidden,
            guidance_dim=guidance_dim,
            comm_radius=comm_radius,
            num_heads=num_heads,
        )
    if mode in ("full", "dsfg"):
        return DSGF(
            obs_dim=obs_dim,
            hidden_dim=hidden,
            guidance_dim=guidance_dim,
            comm_radius=comm_radius,
            num_heads=num_heads,
            num_spatial_layers=cfg.get("num_spatial_layers", 2),
            use_dynamic_quality=cfg.get("use_dynamic_quality", True),
            use_temporal=cfg.get("use_temporal", True),
        )
    if mode in ("ac_dsgf", "ac-dsgf"):
        return ACDSGF(
            obs_dim=obs_dim,
            hidden_dim=hidden,
            guidance_dim=guidance_dim,
            comm_radius=comm_radius,
            num_heads=num_heads,
            num_spatial_layers=cfg.get("num_spatial_layers", 2),
            use_temporal=cfg.get("use_temporal", True),
            lambda_c=cfg.get("lambda_comm", cfg.get("lambda_c", 1e-3)),
        )
    if mode in ("ac_dsgf_pp", "ac-dsgf-pp", "ac_dsgf++"):
        return ACDSGFpp(
            obs_dim=obs_dim,
            hidden_dim=hidden,
            guidance_dim=guidance_dim,
            comm_radius=comm_radius,
            num_heads=num_heads,
            num_spatial_layers=cfg.get("num_spatial_layers", 2),
            use_temporal=cfg.get("use_temporal", True),
            lambda_c=cfg.get("lambda_comm", cfg.get("lambda_c", 1e-4)),
            gate_residual=cfg.get("gate_residual", True),
            gate_residual_alpha=float(cfg.get("gate_residual_alpha", 0.5)),
            use_utility_in_gate=bool(cfg.get("use_utility_in_gate", True)),
            use_utility_in_policy=bool(cfg.get("use_utility_in_policy", True)),
            utility_target=str(cfg.get("utility_target", "action")),
            cau_alpha=float(cfg.get("cau_alpha", 0.5)),
            cau_beta=float(cfg.get("cau_beta", 0.4)),
            cau_gamma=float(cfg.get("cau_gamma", 0.1)),
            cau_horizon=int(cfg.get("cau_horizon", 10)),
            cau_td_gamma=float(cfg.get("cau_td_gamma", 0.95)),
        )
    raise ValueError(f"Unknown guidance mode: {mode}")
