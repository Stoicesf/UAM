"""Wireless channel: h = α / d^κ + ξ  (path-loss core; ξ via optional noise)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ChannelConfig:
    bandwidth_hz: float = 1.0
    tx_power: float = 1.0
    noise: float = 0.05
    path_loss_exp: float = 2.2  # κ
    path_loss_gain: float = 1.0  # α
    comm_radius: float = 2.5
    eps: float = 1e-6
    shadow_std: float = 0.0  # ξ scale (0 = deterministic)


def pairwise_distance(pos: torch.Tensor) -> torch.Tensor:
    d = pos.unsqueeze(2) - pos.unsqueeze(1)
    return torch.linalg.norm(d, dim=-1).clamp(min=0.0)


def path_gain(dist: torch.Tensor, cfg: ChannelConfig) -> torch.Tensor:
    """h_ij = α / (d^κ + eps); diagonal zeroed. Optional multiplicative ξ."""
    h = cfg.path_loss_gain / (dist.pow(cfg.path_loss_exp) + cfg.eps)
    if cfg.shadow_std > 0:
        xi = torch.randn_like(h) * cfg.shadow_std
        h = h * torch.exp(xi)
    eye = torch.eye(dist.shape[-1], device=dist.device, dtype=torch.bool)
    return h.masked_fill(eye.unsqueeze(0), 0.0)


def link_sinr(
    pos: torch.Tensor,
    cfg: ChannelConfig | None = None,
    tx_power: torch.Tensor | None = None,
) -> torch.Tensor:
    cfg = cfg or ChannelConfig()
    dist = pairwise_distance(pos)
    h = path_gain(dist, cfg)
    B, N, _ = pos.shape
    if tx_power is None:
        P = torch.full((B, N), cfg.tx_power, device=pos.device, dtype=pos.dtype)
    else:
        P = tx_power
        if P.dim() == 1:
            P = P.unsqueeze(0).expand(B, -1)
    rx = P.unsqueeze(-1) * h
    total_at_j = rx.sum(dim=1)
    interfer = total_at_j.unsqueeze(1) - rx
    sinr = rx / (cfg.noise + interfer.clamp(min=0.0) + cfg.eps)
    mask = (dist > 0) & (dist <= cfg.comm_radius)
    return sinr * mask.float()


def observe_channel(pos: torch.Tensor, cfg: ChannelConfig | None = None) -> dict[str, torch.Tensor]:
    """
    Returns geometry-channel pack used by teacher:
      position, channel_gain, interference, sinr (link), sinr_bar
    """
    cfg = cfg or ChannelConfig()
    dist = pairwise_distance(pos)
    h = path_gain(dist, cfg)
    sinr = link_sinr(pos, cfg)
    active = sinr > 0
    denom = active.float().sum(dim=(1, 2)).clamp(min=1.0)
    sinr_bar = (sinr * active.float()).sum(dim=(1, 2)) / denom
    h_bar = (h * active.float()).sum(dim=(1, 2)) / denom
    rx = cfg.tx_power * h
    total_at_j = rx.sum(dim=1)
    interfer = (total_at_j.unsqueeze(1) - rx).clamp(min=0.0)
    I_bar = (interfer * active.float()).sum(dim=(1, 2)) / denom
    return {
        "position": pos,
        "channel_gain": h_bar,
        "interference": I_bar,
        "sinr": sinr,
        "sinr_bar": sinr_bar,
        "dist": dist,
        "h": h,
    }
