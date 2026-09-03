"""Anchor election for lightweight relative localization."""

from __future__ import annotations

from enum import Enum

import torch


class AnchorStrategy(str, Enum):
    CONFIDENCE = "confidence"
    BATTERY = "battery"
    GEOMETRY = "geometry"


def select_anchors(
    positions: torch.Tensor,
    *,
    n_anchors: int = 2,
    strategy: AnchorStrategy | str = AnchorStrategy.GEOMETRY,
    confidence: torch.Tensor | None = None,
    battery: torch.Tensor | None = None,
    prev_anchors: torch.Tensor | None = None,
    switch_hysteresis: float = 0.05,
) -> torch.Tensor:
    """Return (B, K) long indices of elected anchors.

    strategies:
      confidence — highest position confidence
      battery — highest remaining battery
      geometry — maximize pairwise spread (greedy farthest)
    """
    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    b, n, _ = positions.shape
    k = int(max(1, min(n_anchors, n)))
    strategy = AnchorStrategy(strategy)

    if strategy == AnchorStrategy.CONFIDENCE:
        if confidence is None:
            confidence = torch.ones(b, n, device=positions.device)
        scores = confidence
    elif strategy == AnchorStrategy.BATTERY:
        if battery is None:
            battery = torch.ones(b, n, device=positions.device)
        scores = battery
    else:
        # geometry: score = mean distance to others (prefer peripheral spread starters)
        dist = (positions.unsqueeze(2) - positions.unsqueeze(1)).norm(dim=-1)
        scores = dist.sum(dim=-1) / max(n - 1, 1)

    # top-k by score
    _, idx = torch.topk(scores, k=k, dim=-1)

    # hysteresis: keep prev if overlap large (simulates ≤500ms switch stickiness)
    if prev_anchors is not None and prev_anchors.shape == idx.shape:
        # if mean score drop from keeping prev is small, keep
        gather_prev = torch.gather(scores, 1, prev_anchors.clamp(0, n - 1))
        gather_new = torch.gather(scores, 1, idx)
        keep = (gather_prev.mean(dim=-1) + switch_hysteresis) >= gather_new.mean(dim=-1)
        idx = torch.where(keep.unsqueeze(-1), prev_anchors, idx)
    return idx


def self_check() -> None:
    pos = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
    a = select_anchors(pos, n_anchors=2, strategy="geometry")
    assert a.shape == (1, 2)
    bat = torch.tensor([[0.1, 0.9, 0.2, 0.3]])
    b = select_anchors(pos, n_anchors=1, strategy="battery", battery=bat)
    assert int(b[0, 0]) == 1
    c = select_anchors(
        pos, n_anchors=2, strategy="confidence", confidence=torch.tensor([[1, 0, 0, 0.5]])
    )
    assert 0 in set(c[0].tolist())
    print("anchor_selector: OK")


if __name__ == "__main__":
    self_check()
