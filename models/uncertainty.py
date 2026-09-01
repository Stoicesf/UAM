"""Packet loss / node communication failure for AC-DSGF evaluation & Demo."""

from __future__ import annotations

import torch


def apply_packet_loss(
    g: torch.Tensor,
    loss_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Independently drop each edge with probability loss_prob."""
    if loss_prob <= 0:
        return g
    if loss_prob >= 1:
        return torch.zeros_like(g)
    keep = torch.bernoulli(
        torch.full_like(g, 1.0 - loss_prob),
        generator=generator,
    )
    # keep self-mask: already zero on diagonal typically
    return g * keep


def silence_agent(g: torch.Tensor, agent_index: int) -> torch.Tensor:
    """Zero all outgoing and incoming gates for one UAV (comm failure)."""
    out = g.clone()
    out[..., agent_index, :] = 0
    out[..., :, agent_index] = 0
    return out
