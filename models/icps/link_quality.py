"""Map SINR/SNR to edge quality q_ij and soft packet-loss probability."""

from __future__ import annotations

import torch


def snr_to_quality(
    snr_db: torch.Tensor,
    *,
    snr_min: float = 0.0,
    snr_max: float = 20.0,
) -> torch.Tensor:
    """Piecewise-linear map SNR dB → q in [0, 1]."""
    q = (snr_db - snr_min) / max(snr_max - snr_min, 1e-6)
    return q.clamp(0.0, 1.0)


def sinr_to_quality(
    sinr_db: torch.Tensor,
    *,
    sinr_min: float = 0.0,
    sinr_max: float = 20.0,
) -> torch.Tensor:
    return snr_to_quality(sinr_db, snr_min=sinr_min, snr_max=sinr_max)


def soft_packet_loss_prob(
    snr_or_sinr_db: torch.Tensor,
    *,
    snr_good: float = 20.0,
    snr_bad: float = 0.0,
) -> torch.Tensor:
    """Higher loss when link is weak: p_loss = 1 - q."""
    q = snr_to_quality(snr_or_sinr_db, snr_min=snr_bad, snr_max=snr_good)
    return (1.0 - q).clamp(0.0, 1.0)


def apply_channel_packet_loss(
    g: torch.Tensor,
    snr_or_sinr_db: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Drop edges with per-edge Bernoulli(p_loss(SNR)). Fusion point #5."""
    p = soft_packet_loss_prob(snr_or_sinr_db)
    if generator is None:
        keep = torch.bernoulli(1.0 - p)
    else:
        keep = torch.bernoulli(1.0 - p, generator=generator)
    return g * keep


def bandwidth_to_budget_ratio(
    bandwidth_hz: float,
    *,
    bandwidth_ref_hz: float = 20e6,
    min_ratio: float = 0.1,
    max_ratio: float = 1.0,
) -> float:
    """Map physical bandwidth quota → Π_Bt budget_ratio (fusion point #2)."""
    r = bandwidth_hz / max(bandwidth_ref_hz, 1.0)
    return float(max(min_ratio, min(max_ratio, r)))


def self_check() -> None:
    snr = torch.tensor([[0.0, 10.0, 20.0]])
    q = snr_to_quality(snr)
    assert torch.allclose(q, torch.tensor([[0.0, 0.5, 1.0]]), atol=1e-5)
    p = soft_packet_loss_prob(snr)
    assert torch.allclose(p, 1 - q)
    g = torch.ones(1, 3, 3)
    g2 = apply_channel_packet_loss(g, snr.unsqueeze(-1).expand(1, 3, 3) * 0 + 20)
    assert g2.shape == g.shape
    assert abs(bandwidth_to_budget_ratio(10e6) - 0.5) < 1e-9
    print("link_quality: OK")


if __name__ == "__main__":
    self_check()
