"""Eval-only communication channel impairments (T-RO §6.5).

Does not train. Mutates forward paths via temporary attributes / hooks.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
import torch.nn as nn

from guidance.factory import SparseGuideEncoder
from models.ac_dsgf import ACDSGF
from models.dsgf import DSGF
from models.guide_gnn import GATGuideEncoder


def clear_channel(module: nn.Module) -> None:
    for name in (
        "channel_packet_loss",
        "channel_delay_steps",
        "packet_loss_p",  # legacy RA-L hook name
    ):
        if hasattr(module, name):
            setattr(module, name, 0.0 if "loss" in name else 0)
    if hasattr(module, "_channel_gate_buf"):
        module._channel_gate_buf = None  # type: ignore[attr-defined]


def set_channel(
    module: nn.Module,
    *,
    packet_loss: float = 0.0,
    delay_steps: int = 0,
) -> None:
    module.channel_packet_loss = float(packet_loss)  # type: ignore[attr-defined]
    module.channel_delay_steps = int(delay_steps)  # type: ignore[attr-defined]
    module.packet_loss_p = float(packet_loss)  # type: ignore[attr-defined]  # compat
    if delay_steps <= 0 and hasattr(module, "_channel_gate_buf"):
        module._channel_gate_buf = None  # type: ignore[attr-defined]


def _apply_loss(g: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return g
    return g * (torch.rand_like(g) > p).float()


def _apply_delay(module: nn.Module, g: torch.Tensor, delay: int) -> torch.Tensor:
    if delay <= 0:
        return g
    buf: deque | None = getattr(module, "_channel_gate_buf", None)
    if buf is None:
        buf = deque(maxlen=delay)
        module._channel_gate_buf = buf  # type: ignore[attr-defined]
    if len(buf) < delay:
        # warm-up: use zeros (no stale message yet)
        out = torch.zeros_like(g)
    else:
        out = buf[0]
    buf.append(g.detach())
    return out


def install_channel_hooks() -> None:
    """Idempotent monkeypatches for eval-only packet loss + delay."""
    if getattr(install_channel_hooks, "_done", False):
        return

    _ac = ACDSGF.forward

    def ac_fwd(self, obs, positions=None, z_prev=None, apply_budget=False):
        phi, g, diag = _ac(self, obs, positions, z_prev, apply_budget)
        if self.training:
            return phi, g, diag
        p = float(getattr(self, "channel_packet_loss", 0.0) or 0.0)
        d = int(getattr(self, "channel_delay_steps", 0) or 0)
        if p <= 0 and d <= 0:
            return phi, g, diag
        # Re-apply channel on gates and recompute aggregation path lightly:
        # full recompute via controller is expensive; zero dropped edges in g / A_t.
        g2 = _apply_loss(g, p)
        g2 = _apply_delay(self, g2, d)
        # Update diagnostics to reflect delivered topology
        from models.communication.budget_layer import hard_adjacency, degree_budget_violation

        A_hard = hard_adjacency(g2)
        radius_mask = diag.get("radius_mask")
        if radius_mask is None:
            return phi, g2, diag
        k_fixed = self.fixed_k
        ratio = self.budget_ratio
        feas = degree_budget_violation(
            A_hard,
            radius_mask,
            budget_ratio=None if k_fixed is not None else (float(ratio) if ratio is not None else 1.0),
            k_fixed=int(k_fixed) if k_fixed is not None else None,
        )
        diag = dict(diag)
        diag["g"] = g2
        diag["A_t"] = A_hard
        diag["C_t"] = torch.tensor(feas["C"], device=g2.device)
        diag["mean_degree"] = torch.tensor(feas["mean_degree"], device=g2.device)
        diag["rho_t"] = torch.tensor(feas["rho"], device=g2.device)
        diag["degrees"] = feas["degrees"]
        diag["channel_packet_loss"] = torch.tensor(p, device=g2.device)
        diag["channel_delay_steps"] = torch.tensor(float(d), device=g2.device)
        # Note: phi was computed with pre-channel g; for stress-test we optionally
        # re-run spatial with impaired A. Prefer consistency of control under loss:
        # recompute phi from impaired A_tilde when channel is active.
        if positions is None:
            positions = obs[..., :2] if obs.dim() >= 2 else None
        if positions is not None and obs is not None:
            obs_b = obs.unsqueeze(0) if obs.dim() == 2 else obs
            pos_b = positions.unsqueeze(0) if positions.dim() == 2 else positions
            h = self.node_embed(obs_b)
            q_ij = diag.get("q")
            if q_ij is None:
                return phi, g2, diag
            A_tilde = radius_mask * q_ij * g2
            x = h
            for layer in self.spatial_layers:
                x = torch.tanh(layer(x, A_tilde) + x)
            if self.temporal is not None:
                z = self.temporal(x, z_prev)
            else:
                z = x
            phi2 = self.phi_head(z)
            direction = phi2[..., :2]
            phi2 = phi2.clone()
            phi2[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            diag["A_tilde"] = A_tilde
            return phi2, g2, diag
        return phi, g2, diag

    ACDSGF.forward = ac_fwd  # type: ignore[assignment]

    _dsgf = DSGF.forward

    def dsgf_fwd(self, obs, positions=None, comm_mask=None, z_prev=None):
        p = float(getattr(self, "channel_packet_loss", 0.0) or 0.0)
        d = int(getattr(self, "channel_delay_steps", 0) or 0)
        if (p <= 0 and d <= 0) or self.training:
            return _dsgf(self, obs, positions, comm_mask, z_prev)
        dg = self.dynamic_graph
        _dg = dg.forward

        def noisy(*a, **k):
            wadj, q = _dg(*a, **k)
            wadj = _apply_loss(wadj, p)
            wadj = _apply_delay(self, wadj, d)
            q = q * (wadj > 0).float()
            return wadj, q

        dg.forward = noisy  # type: ignore
        try:
            return _dsgf(self, obs, positions, comm_mask, z_prev)
        finally:
            dg.forward = _dg  # type: ignore

    DSGF.forward = dsgf_fwd  # type: ignore[assignment]

    _sparse = SparseGuideEncoder.forward

    def sparse_fwd(self, obs: torch.Tensor) -> torch.Tensor:
        p = float(getattr(self, "channel_packet_loss", 0.0) or 0.0)
        d = int(getattr(self, "channel_delay_steps", 0) or 0)
        if (p <= 0 and d <= 0) or self.training:
            return _sparse(self, obs)
        x = self.proj(obs)
        b, n, _ = x.shape
        adj = x.new_ones(b, n, n)
        eye = torch.eye(n, device=x.device).unsqueeze(0)
        adj = adj * (1.0 - eye)
        adj = _apply_loss(adj, p)
        adj = _apply_delay(self, adj, d)
        # Avoid empty graph → NaN attention: restore uniform links for isolated rows
        row_empty = adj.sum(dim=-1) <= 0  # (B, N)
        if bool(row_empty.any()):
            fill = (1.0 - eye) / max(n - 1, 1)
            adj = torch.where(row_empty.unsqueeze(-1), fill, adj)
        x = self.attn(x, adj)
        phi = self.head(x)
        direction = phi[..., :2]
        phi = phi.clone()
        phi[..., :2] = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.nan_to_num(phi, nan=0.0)

    SparseGuideEncoder.forward = sparse_fwd  # type: ignore[assignment]

    # GAT: reuse loss/delay on adjacency
    _gat = GATGuideEncoder.forward

    def gat_fwd(self, obs, positions=None):
        p = float(getattr(self, "channel_packet_loss", 0.0) or 0.0)
        d = int(getattr(self, "channel_delay_steps", 0) or 0)
        if (p <= 0 and d <= 0) or self.training:
            return _gat(self, obs, positions)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
        if positions is None:
            positions = obs[..., :2]
        elif positions.dim() == 2:
            positions = positions.unsqueeze(0)
        from guidance.graph_builder import build_adjacency

        adj = build_adjacency(positions, self.comm_radius)
        adj = _apply_loss(adj, p)
        adj = _apply_delay(self, adj, d)
        h = torch.tanh(self.input_proj(obs))
        for layer in self.gat_layers:
            h = torch.tanh(layer(h, adj) + h)
        return self.guidance_head(h)

    GATGuideEncoder.forward = gat_fwd  # type: ignore[assignment]
    install_channel_hooks._done = True  # type: ignore[attr-defined]


def set_policy_channel(policy: nn.Module, *, packet_loss: float = 0.0, delay_steps: int = 0) -> None:
    """Set channel attrs on all known encoders inside a policy."""
    from algorithms.guided.mappo_guided import ACGuideAdapter, GraphGuideAdapter

    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            set_channel(m.encoder, packet_loss=packet_loss, delay_steps=delay_steps)
        elif isinstance(m, GraphGuideAdapter):
            set_channel(m.encoder, packet_loss=packet_loss, delay_steps=delay_steps)
        elif isinstance(m, (ACDSGF, DSGF, GATGuideEncoder, SparseGuideEncoder)):
            set_channel(m, packet_loss=packet_loss, delay_steps=delay_steps)


def clear_policy_channel(policy: nn.Module) -> None:
    from algorithms.guided.mappo_guided import ACGuideAdapter, GraphGuideAdapter

    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            clear_channel(m.encoder)
        elif isinstance(m, GraphGuideAdapter):
            clear_channel(m.encoder)
        elif isinstance(m, (ACDSGF, DSGF, GATGuideEncoder, SparseGuideEncoder)):
            clear_channel(m)
