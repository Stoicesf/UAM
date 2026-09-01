"""Canonical capacity teacher — ONLY place that defines c_t.

    c_t = W * log2(1 + SINR_bar)

All experiments / training MUST import from here (or via pack_state).
"""

from __future__ import annotations

import torch

from secdo.datasets.uav.channel import ChannelConfig, link_sinr, observe_channel


def capacity_teacher(
    pos: torch.Tensor,
    cfg: ChannelConfig | None = None,
    tx_power: torch.Tensor | None = None,
    reduce: str = "mean_link",
) -> torch.Tensor:
    """
    Teacher c_t = W log2(1 + SINR_agg).

    Returns:
        c: [B, 1]
    """
    cfg = cfg or ChannelConfig()
    sinr = link_sinr(pos, cfg, tx_power)
    active = sinr > 0
    if reduce == "mean_link":
        denom = active.float().sum(dim=(1, 2)).clamp(min=1.0)
        sinr_bar = (sinr * active.float()).sum(dim=(1, 2)) / denom
        c = cfg.bandwidth_hz * torch.log2(1.0 + sinr_bar)
    elif reduce == "sum_rate":
        rates = cfg.bandwidth_hz * torch.log2(1.0 + sinr)
        c = rates.sum(dim=(1, 2)) / float(sinr.shape[-1])
    else:
        raise ValueError(reduce)
    return c.unsqueeze(-1).clamp(min=cfg.eps)


def pack_state(
    pos: torch.Tensor,
    vel: torch.Tensor | None = None,
    energy: torch.Tensor | None = None,
    cfg: ChannelConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Full observation pack; c_teacher always from capacity_teacher()."""
    cfg = cfg or ChannelConfig()
    B, N, D = pos.shape
    device, dtype = pos.device, pos.dtype
    if vel is None:
        vel = torch.zeros(B, N, D, device=device, dtype=dtype)
    if energy is None:
        energy = torch.ones(B, N, device=device, dtype=dtype)

    ch = observe_channel(pos, cfg)
    c_teacher = capacity_teacher(pos, cfg)

    mean_pos = pos.mean(dim=1)[..., :2]
    if mean_pos.shape[-1] < 2:
        mean_pos = torch.nn.functional.pad(mean_pos, (0, 2 - mean_pos.shape[-1]))
    speed = vel.norm(dim=-1).mean(dim=-1, keepdim=True)
    e_bar = energy.mean(dim=-1, keepdim=True)
    feat = torch.cat(
        [
            mean_pos[..., :2],
            speed,
            ch["channel_gain"].unsqueeze(-1),
            ch["interference"].unsqueeze(-1),
            e_bar,
            ch["sinr_bar"].unsqueeze(-1),
        ],
        dim=-1,
    )
    return {
        "position": pos,
        "velocity": vel,
        "channel_gain": ch["channel_gain"],
        "interference": ch["interference"],
        "remaining_energy": e_bar.squeeze(-1),
        "sinr_bar": ch["sinr_bar"],
        "sinr": ch["sinr"],
        "c_teacher": c_teacher,
        "feat": feat,
    }


# Back-compat aliases used by legacy models.secdo imports
system_capacity_teacher = capacity_teacher
build_state_features = pack_state
UAVChannelConfig = ChannelConfig
