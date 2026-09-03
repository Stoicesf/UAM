"""GDOP (geometric dilution of precision) for relative ranging."""

from __future__ import annotations

import torch


def gdop_from_anchors(
    target: torch.Tensor,
    anchors: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """2D GDOP for one target vs K anchors.

    target: (B, 2) or (2,)
    anchors: (B, K, 2) or (K, 2)
    returns: (B,) GDOP
    """
    if target.dim() == 1:
        target = target.unsqueeze(0)
    if anchors.dim() == 2:
        anchors = anchors.unsqueeze(0)
    # unit LOS vectors
    diff = anchors - target.unsqueeze(1)  # (B,K,2)
    dist = diff.norm(dim=-1, keepdim=True).clamp(min=eps)
    H = diff / dist  # (B,K,2)
    # GDOP = sqrt(tr((H^T H)^{-1}))
    HtH = H.transpose(-1, -2) @ H  # (B,2,2)
    # add jitter for rank deficiency
    eye = torch.eye(2, device=target.device, dtype=target.dtype).unsqueeze(0)
    HtH = HtH + eps * eye
    inv = torch.linalg.inv(HtH)
    return torch.sqrt(inv.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=0))


def gdop_swarm(
    positions: torch.Tensor,
    anchor_idx: torch.Tensor,
) -> torch.Tensor:
    """Per-agent GDOP given anchor indices. Anchors themselves get GDOP=0."""
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    b, n, _ = positions.shape
    k = anchor_idx.shape[-1]
    out = torch.zeros(b, n, device=positions.device, dtype=positions.dtype)
    for bi in range(b):
        anch = positions[bi, anchor_idx[bi]]  # (K,2)
        for i in range(n):
            if i in set(anchor_idx[bi].tolist()):
                out[bi, i] = 0.0
                continue
            out[bi, i] = gdop_from_anchors(positions[bi, i], anch.unsqueeze(0))
    _ = k
    return out


def self_check() -> None:
    # equilateral-ish anchors → finite GDOP
    target = torch.tensor([0.5, 0.5])
    anchors = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
    g = gdop_from_anchors(target, anchors)
    assert g.ndim == 1 and float(g[0]) > 0
    print(f"gdop: OK (GDOP={float(g[0]):.3f})")


if __name__ == "__main__":
    self_check()
