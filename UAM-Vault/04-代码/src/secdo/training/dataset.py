"""Offline UAV trajectory dataset for Stage I pretraining."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig, UAVBandwidthEnv


@dataclass
class TrajectoryBatch:
    feat: torch.Tensor  # [T+1, B, F] stored as list of episodes flattened later
    c: torch.Tensor


class UAVTrajectoryDataset(Dataset):
    """
    Pre-rolls teacher trajectories offline.
    Each item: dict with feat[T,F], c[T,1] on CPU (moved to device in train loop).
    """

    def __init__(
        self,
        env_cfg: UAVBandwidthConfig,
        n_traj: int = 1000,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        self.env_cfg = env_cfg
        self.n_traj = n_traj
        self.seed = seed
        self.samples: list[dict[str, torch.Tensor]] = []
        self._generate(device)

    def _generate(self, device: str | torch.device) -> None:
        env = UAVBandwidthEnv(self.env_cfg, device=device)
        for i in range(self.n_traj):
            env.cfg.seed = self.seed + i
            obs = env.reset(batch=1)
            feats = [obs["feat"].squeeze(0).cpu()]
            cs = [obs["c_teacher"].squeeze(0).cpu()]
            while not env.done:
                obs = env.step()
                feats.append(obs["feat"].squeeze(0).cpu())
                cs.append(obs["c_teacher"].squeeze(0).cpu())
            self.samples.append(
                {
                    "feat": torch.stack(feats, dim=0),  # [T+1, F]
                    "c": torch.stack(cs, dim=0),  # [T+1, 1] or [T+1]
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_pad(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack equal-horizon trajectories (env horizon fixed)."""
    feat = torch.stack([b["feat"] for b in batch], dim=1)  # [T, B, F]
    c = torch.stack([b["c"] for b in batch], dim=1)
    if c.dim() == 2:
        c = c.unsqueeze(-1)
    return {"feat": feat, "c": c}
