"""Checkpoint save/load — shared by MAPPO and SECDO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def save_best_last(ckpt_dir: str | Path, payload: dict[str, Any], *, is_best: bool) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(ckpt_dir / "last.pt", payload)
    if is_best:
        save_checkpoint(ckpt_dir / "best.pt", payload)
