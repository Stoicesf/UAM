"""Communication interpretability helpers (eval-only; no training).

Soft Comm. Mass C = Σ g_ij is an activation intensity proxy.
Active edge ratio makes the sparsity claim physically readable:
    ρ_edge = |{g_ij > τ} ∩ {A_ij=1}| / |{A_ij=1}|
"""

from __future__ import annotations

import torch


def active_edge_ratio(
    gate: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Fraction of radius-feasible edges with gate above threshold."""
    g = gate.float()
    m = mask.float()
    if g.dim() == 2:
        g = g.unsqueeze(0)
        m = m.unsqueeze(0)
    active = ((g > threshold).float() * m).sum(dim=(-2, -1))
    total = m.sum(dim=(-2, -1)).clamp_min(1.0)
    return float((active / total).mean().item())


def soft_comm_mass(gate: torch.Tensor) -> float:
    g = gate.float()
    if g.dim() == 2:
        return float(g.sum().item())
    return float(g.sum(dim=(-2, -1)).mean().item())


def mass_ratio_on_support(gate: torch.Tensor, mask: torch.Tensor) -> float:
    """C / |E_radius| = mean soft activation on feasible edges (incl. near-zero)."""
    g = gate.float()
    m = mask.float()
    if g.dim() == 2:
        g = g.unsqueeze(0)
        m = m.unsqueeze(0)
    mass = (g * m).sum(dim=(-2, -1))
    total = m.sum(dim=(-2, -1)).clamp_min(1.0)
    return float((mass / total).mean().item())


def radius_mask_from_positions(
    positions: torch.Tensor,
    comm_radius: float,
) -> torch.Tensor:
    """positions: (B,N,2) or (N,2) → binary mask with zero diagonal."""
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
    n = positions.shape[-2]
    eye = torch.eye(n, device=positions.device, dtype=positions.dtype).unsqueeze(0)
    return ((dist < comm_radius).float() * (1.0 - eye))


def expected_packet_survival(loss_rate: float) -> float:
    """Independent Bernoulli drop model: E[recv | active] = 1 - p_loss."""
    return float(max(0.0, min(1.0, 1.0 - loss_rate)))


def packet_survival_on_active(
    gate: torch.Tensor,
    mask: torch.Tensor,
    loss_rate: float,
    threshold: float = 0.5,
) -> float:
    """Expected received fraction among active edges under i.i.d. packet loss."""
    g = gate.float()
    m = mask.float()
    if g.dim() == 2:
        g = g.unsqueeze(0)
        m = m.unsqueeze(0)
    active = (g > threshold).float() * m
    n_act = active.sum(dim=(-2, -1)).clamp_min(1e-8)
    # each active edge kept with prob (1-loss)
    recv = active * (1.0 - loss_rate)
    return float((recv.sum(dim=(-2, -1)) / n_act).mean().item())
