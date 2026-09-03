"""Abstract dual-channel ICPS physical-layer model for VMAS.

COMM_POS_CHANNEL — joint communication + ranging
SENSE_CHANNEL    — sensing / perception waveform occupancy

Not a full phased-array waveform sim (see ICPS feasibility note).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


@dataclass
class ChannelConfig:
    path_loss_exp: float = 2.0
    noise_floor_dbm: float = -90.0
    interference_thresh_db: float = 0.0
    bandwidth_hz: float = 20e6
    tx_power_dbm: float = 20.0
    carrier_freq_hz: float = 2.4e9
    shadow_sigma_db: float = 4.0
    two_ray_ht_m: float = 1.5
    two_ray_hr_m: float = 1.5
    model: Literal["fspl", "two_ray"] = "fspl"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelConfig":
        keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in keys})


def _dbm_to_mw(dbm: torch.Tensor | float) -> torch.Tensor:
    if not isinstance(dbm, torch.Tensor):
        dbm = torch.tensor(float(dbm))
    return 10.0 ** (dbm / 10.0)


def free_space_path_loss_db(
    dist_m: torch.Tensor,
    carrier_freq_hz: float,
    path_loss_exp: float = 2.0,
) -> torch.Tensor:
    """FSPL-style: PL(d) = 20log10(d) + 20log10(f) - 147.55, with tunable exponent.

    When path_loss_exp != 2, replace 20 with 10*exp on the distance term.
    """
    d = dist_m.clamp(min=1e-3)
    # reference FSPL at exponent 2
    pl = (
        10.0 * path_loss_exp * torch.log10(d)
        + 20.0 * torch.log10(torch.tensor(carrier_freq_hz, device=d.device, dtype=d.dtype))
        - 147.55
    )
    return pl


def two_ray_path_loss_db(
    dist_m: torch.Tensor,
    ht_m: float,
    hr_m: float,
    carrier_freq_hz: float,
) -> torch.Tensor:
    """Two-ray ground reflection approximation (breakpoint crossover)."""
    d = dist_m.clamp(min=1e-3)
    d_break = 4.0 * ht_m * hr_m * carrier_freq_hz / 3e8
    # near: FSPL; far: ~40 log10(d) - 20 log10(ht*hr)
    pl_near = free_space_path_loss_db(d, carrier_freq_hz, path_loss_exp=2.0)
    pl_far = (
        40.0 * torch.log10(d)
        - 20.0 * torch.log10(torch.tensor(ht_m * hr_m, device=d.device, dtype=d.dtype))
    )
    return torch.where(d < d_break, pl_near, pl_far)


class DualChannelModel:
    """Pairwise SNR/SINR on COMM_POS and occupancy on SENSE."""

    def __init__(self, cfg: ChannelConfig | None = None):
        self.cfg = cfg or ChannelConfig()
        self._gen = torch.Generator()
        self._gen.manual_seed(int(self.cfg.seed))

    def path_loss_db(self, dist_m: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if c.model == "two_ray":
            return two_ray_path_loss_db(
                dist_m, c.two_ray_ht_m, c.two_ray_hr_m, c.carrier_freq_hz
            )
        return free_space_path_loss_db(dist_m, c.carrier_freq_hz, c.path_loss_exp)

    def shadowing_db(self, like: torch.Tensor) -> torch.Tensor:
        if self.cfg.shadow_sigma_db <= 0:
            return torch.zeros_like(like)
        # ponytail: i.i.d. shadow per edge; upgrade = spatial correlated field
        return torch.randn(
            like.shape, generator=self._gen, dtype=like.dtype, device="cpu"
        ).to(like.device) * self.cfg.shadow_sigma_db

    def rx_power_dbm(self, dist_m: torch.Tensor) -> torch.Tensor:
        pl = self.path_loss_db(dist_m) + self.shadowing_db(dist_m)
        return self.cfg.tx_power_dbm - pl

    def snr_db(self, dist_m: torch.Tensor) -> torch.Tensor:
        return self.rx_power_dbm(dist_m) - self.cfg.noise_floor_dbm

    def sinr_db(
        self,
        dist_m: torch.Tensor,
        active_tx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Same-channel interference from concurrent transmitters.

        dist_m: (B, N, N) pairwise distances (meters).
        active_tx: (B, N) who transmits (default: all).
        SINR_ij = Pr(i←j) / (N0 + sum_{k≠i,j} Pr(i←k) a_k)
        """
        if dist_m.dim() == 2:
            dist_m = dist_m.unsqueeze(0)
        b, n, _ = dist_m.shape
        pr_mw = _dbm_to_mw(self.rx_power_dbm(dist_m))
        noise_mw = _dbm_to_mw(
            torch.tensor(self.cfg.noise_floor_dbm, device=dist_m.device, dtype=dist_m.dtype)
        )
        if active_tx is None:
            active_tx = torch.ones(b, n, device=dist_m.device, dtype=dist_m.dtype)
        elif active_tx.dim() == 1:
            active_tx = active_tx.unsqueeze(0)

        eye = torch.eye(n, device=dist_m.device, dtype=dist_m.dtype).unsqueeze(0)
        a_k = active_tx.unsqueeze(1).expand(b, n, n)
        # sum_{k≠i} Pr_ik a_k, then subtract desired j
        total = (pr_mw * a_k * (1.0 - eye)).sum(dim=-1, keepdim=True)
        I = (total - pr_mw * a_k).clamp(min=0.0)
        sig = pr_mw * a_k
        sinr = sig / (I + noise_mw + 1e-18)
        sinr_db = 10.0 * torch.log10(sinr.clamp(min=1e-18))
        return sinr_db * (1.0 - eye) - 1e6 * eye

    def sense_occupancy(
        self,
        dist_m: torch.Tensor,
        sense_active: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SENSE_CHANNEL resource occupancy in [0,1] from concurrent sensors."""
        if dist_m.dim() == 2:
            dist_m = dist_m.unsqueeze(0)
        b, n, _ = dist_m.shape
        if sense_active is None:
            sense_active = torch.ones(b, n, device=dist_m.device, dtype=dist_m.dtype)
        # fraction of others sensing within radius proxy
        near = (dist_m < 50.0).float()
        eye = torch.eye(n, device=dist_m.device).unsqueeze(0)
        near = near * (1.0 - eye)
        occ = (near * sense_active.unsqueeze(1)).sum(dim=-1) / max(n - 1, 1)
        return occ.clamp(0, 1)

    def conflict(
        self,
        dist_m: torch.Tensor,
        comm_active: torch.Tensor,
        sense_active: torch.Tensor,
    ) -> torch.Tensor:
        """Per-node conflict if both channels heavily loaded (toy MAC)."""
        sinr = self.sinr_db(dist_m, comm_active)
        # mean incoming SINR
        eye = torch.eye(sinr.shape[-1], device=sinr.device).unsqueeze(0)
        valid = 1.0 - eye
        mean_sinr = (sinr * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1)
        occ = self.sense_occupancy(dist_m, sense_active)
        return ((mean_sinr < self.cfg.interference_thresh_db) | (occ > 0.7)).float()


def self_check() -> None:
    cfg = ChannelConfig(shadow_sigma_db=0.0, seed=0)
    m = DualChannelModel(cfg)
    # FSPL vs theory at 100m, 2.4GHz, n=2
    d = torch.tensor(100.0)
    pl = float(m.path_loss_db(d))
    theory = 20 * torch.log10(torch.tensor(100.0)) + 20 * torch.log10(
        torch.tensor(2.4e9)
    ) - 147.55
    err = abs(pl - float(theory)) / abs(float(theory))
    assert err <= 0.05, f"FSPL err {err:.3%} > 5%"

    dist = torch.rand(1, 4, 4) * 100 + 1
    dist = 0.5 * (dist + dist.transpose(-1, -2))
    dist.diagonal(dim1=-2, dim2=-1).fill_(0)
    sinr = m.sinr_db(dist)
    assert sinr.shape == (1, 4, 4)
    print(f"channels: OK (FSPL err={err:.2%})")


if __name__ == "__main__":
    self_check()
