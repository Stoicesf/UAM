"""Histogram-binned mutual information probe (no sklearn)."""

from __future__ import annotations

import math

import torch


def _bin_indices(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Map each feature to [0, n_bins) via per-dim min/max; flatten last dims."""
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(dim=0).values
    hi = flat.max(dim=0).values
    span = (hi - lo).clamp(min=1e-6)
    norm = ((flat - lo) / span).clamp(0, 1 - 1e-6)
    return (norm * n_bins).long()


def _joint_entropy_1d(a: torch.Tensor, b: torch.Tensor, n_bins: int) -> tuple[float, float, float]:
    """H(A), H(B), H(A,B) for integer codes in [0, n_bins) (use first feature only)."""
    aa = a[:, 0].clamp(0, n_bins - 1)
    bb = b[:, 0].clamp(0, n_bins - 1)
    n = aa.numel()
    # joint
    joint = aa * n_bins + bb
    counts = torch.bincount(joint, minlength=n_bins * n_bins).float()
    p = counts / n
    p = p[p > 0]
    hab = float(-(p * p.log()).sum() / math.log(2))
    ca = torch.bincount(aa, minlength=n_bins).float()
    cb = torch.bincount(bb, minlength=n_bins).float()
    pa = ca / n
    pb = cb / n
    pa = pa[pa > 0]
    pb = pb[pb > 0]
    ha = float(-(pa * pa.log()).sum() / math.log(2))
    hb = float(-(pb * pb.log()).sum() / math.log(2))
    return ha, hb, hab


def mutual_info_binned(x: torch.Tensor, z: torch.Tensor, n_bins: int = 10) -> float:
    """I(X; Z) ≈ H(X)+H(Z)-H(X,Z) on first principal dim (binned).

    Uses mean across features of per-dim MI for a cheap multi-dim proxy.
    """
    xb = _bin_indices(x, n_bins)
    zb = _bin_indices(z, n_bins)
    d = min(xb.shape[1], zb.shape[1], 8)
    mis = []
    for i in range(d):
        ha, hb, hab = _joint_entropy_1d(xb[:, i : i + 1], zb[:, i : i + 1], n_bins)
        mis.append(max(0.0, ha + hb - hab))
    return sum(mis) / max(len(mis), 1)


def self_check() -> None:
    torch.manual_seed(0)
    x = torch.randn(500, 4)
    z = x[:, :2] + 0.01 * torch.randn(500, 2)  # near-deterministic
    mi_hi = mutual_info_binned(x[:, :2], z, 10)
    z_noise = torch.randn(500, 2)
    mi_lo = mutual_info_binned(x[:, :2], z_noise, 10)
    assert mi_hi > mi_lo
    print(f"mutual_info_probe: OK (mi_dep={mi_hi:.3f}, mi_indep={mi_lo:.3f})")


if __name__ == "__main__":
    self_check()
