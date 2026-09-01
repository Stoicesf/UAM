"""Dataset loaders for SECDO training (offline trajectories)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_trajectories(root: str | Path) -> list[dict[str, torch.Tensor]]:
    blob = torch.load(Path(root) / "trajectories.pt", map_location="cpu", weights_only=False)
    return blob["trajectories"]


class TrajectoryDataset(Dataset):
    """Wraps generated UAV trajectories (feat, c, rho, position)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = load_manifest(self.root)
        self.samples = load_trajectories(self.root)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_trajectories(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack equal-length trajectories → [T, B, ...]."""
    feat = torch.stack([b["feat"] for b in batch], dim=1)
    c = torch.stack([b["c"] for b in batch], dim=1)
    rho = torch.stack([b["rho"] for b in batch], dim=1)
    return {"feat": feat, "c": c, "rho": rho}
