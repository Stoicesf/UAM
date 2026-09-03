"""EKF relative positioning from COMM_POS ranging + velocity."""

from __future__ import annotations

import torch

from models.icps.channels import ChannelConfig, DualChannelModel
from models.icps.localization.anchor_selector import select_anchors
from models.icps.localization.gdop import gdop_swarm


class RelativePositioningEKF:
    """2D constant-velocity EKF with range measurements to anchors.

    State per agent: [x, y, vx, vy]
    """

    def __init__(
        self,
        n_agents: int,
        *,
        channel: DualChannelModel | None = None,
        process_var: float = 0.05,
        range_var_floor: float = 0.01,
        dt: float = 0.1,
        device: torch.device | None = None,
    ):
        self.n = n_agents
        self.channel = channel or DualChannelModel(ChannelConfig(shadow_sigma_db=0.0))
        self.dt = dt
        self.process_var = process_var
        self.range_var_floor = range_var_floor
        self.device = device or torch.device("cpu")
        self.x = torch.zeros(n_agents, 4, device=self.device)
        self.P = torch.eye(4, device=self.device).unsqueeze(0).repeat(n_agents, 1, 1) * 1.0
        self.anchor_idx: torch.Tensor | None = None

    def reset(self, positions: torch.Tensor, velocities: torch.Tensor | None = None) -> None:
        if positions.dim() == 3:
            positions = positions[0]
        self.x[:, :2] = positions.to(self.device)
        if velocities is not None:
            if velocities.dim() == 3:
                velocities = velocities[0]
            self.x[:, 2:] = velocities.to(self.device)
        self.P = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.n, 1, 1)

    def _range_noise_std(self, snr_db: float) -> float:
        # higher SNR → lower ranging noise; ~cm at 20dB
        q = max(0.0, min(1.0, (snr_db - 0.0) / 20.0))
        return float((1.0 - q) * 2.0 + self.range_var_floor) ** 0.5

    def predict(self) -> None:
        F = torch.eye(4, device=self.device)
        F[0, 2] = self.dt
        F[1, 3] = self.dt
        Q = torch.eye(4, device=self.device) * self.process_var
        self.x = (F @ self.x.unsqueeze(-1)).squeeze(-1)
        self.P = F @ self.P @ F.T + Q

    def update_ranges(
        self,
        true_positions: torch.Tensor,
        anchor_idx: torch.Tensor,
        *,
        snr_db: float = 20.0,
    ) -> None:
        """Synthetic ranging: noisy distance to each anchor (anchors known)."""
        if true_positions.dim() == 3:
            true_positions = true_positions[0]
        if anchor_idx.dim() == 2:
            anchor_idx = anchor_idx[0]
        true_positions = true_positions.to(self.device)
        anchors = true_positions[anchor_idx]
        # anchors get perfect state
        for a in anchor_idx.tolist():
            self.x[a, :2] = true_positions[a]
            self.P[a] = torch.eye(4, device=self.device) * 1e-4

        sigma = self._range_noise_std(snr_db)
        for i in range(self.n):
            if i in set(anchor_idx.tolist()):
                continue
            for a_pos, a_i in zip(anchors, anchor_idx.tolist()):
                true_r = float((true_positions[i] - true_positions[a_i]).norm())
                z = true_r + float(torch.randn(()) * sigma)
                # EKF range update in 2D position subspace
                p = self.x[i, :2]
                diff = p - a_pos
                r_hat = float(diff.norm().clamp(min=1e-4))
                H = torch.zeros(1, 4, device=self.device)
                H[0, 0] = float(diff[0]) / r_hat
                H[0, 1] = float(diff[1]) / r_hat
                R = torch.tensor([[sigma**2]], device=self.device)
                S = H @ self.P[i] @ H.T + R
                K = self.P[i] @ H.T @ torch.linalg.inv(S)
                innov = torch.tensor([z - r_hat], device=self.device)
                self.x[i] = self.x[i] + (K @ innov.unsqueeze(-1)).squeeze(-1)
                I = torch.eye(4, device=self.device)
                self.P[i] = (I - K @ H) @ self.P[i]

    def step(
        self,
        true_positions: torch.Tensor,
        velocities: torch.Tensor,
        *,
        n_anchors: int = 2,
        strategy: str = "geometry",
        snr_db: float = 20.0,
        battery: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if true_positions.dim() == 2:
            true_positions = true_positions.unsqueeze(0)
        if velocities.dim() == 2:
            velocities = velocities.unsqueeze(0)
        anchors = select_anchors(
            true_positions,
            n_anchors=n_anchors,
            strategy=strategy,
            battery=battery,
            confidence=confidence,
            prev_anchors=self.anchor_idx.unsqueeze(0) if self.anchor_idx is not None else None,
        )
        self.anchor_idx = anchors[0]
        self.predict()
        # inject velocity observation lightly
        self.x[:, 2:] = 0.7 * self.x[:, 2:] + 0.3 * velocities[0].to(self.device)
        self.update_ranges(true_positions, anchors, snr_db=snr_db)
        cov_trace = self.P[:, :2, :2].diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        gdop = gdop_swarm(true_positions, anchors)[0]
        return {
            "pos_hat": self.x[:, :2].clone(),
            "vel_hat": self.x[:, 2:].clone(),
            "cov_trace": cov_trace,
            "P": self.P.clone(),
            "anchors": self.anchor_idx.clone(),
            "gdop": gdop,
        }

    def position_uncertainty(self, scale: float = 2.0) -> torch.Tensor:
        """Map cov trace → u_pos in [0,1] for ToA fusion (fusion point #3)."""
        tr = self.P[:, :2, :2].diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return (tr / scale).clamp(0, 1)


def fuse_uncertainty(
    u_epistemic: torch.Tensor,
    u_pos: torch.Tensor,
    mode: str = "max",
) -> torch.Tensor:
    """Combine ensemble u and localization uncertainty. Fixed τ still applies."""
    if mode == "add":
        return (u_epistemic + u_pos).clamp(0, 1)
    if mode == "mean":
        return 0.5 * (u_epistemic + u_pos)
    return torch.maximum(u_epistemic, u_pos)


def self_check() -> None:
    torch.manual_seed(0)
    n = 4
    # square formation
    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    vel = torch.zeros(n, 2)
    ekf = RelativePositioningEKF(n, process_var=0.01)
    ekf.reset(pos + 0.3 * torch.randn(n, 2), vel)
    errs = []
    for _ in range(40):
        out = ekf.step(pos, vel, n_anchors=2, snr_db=20.0)
        errs.append(float((out["pos_hat"] - pos).norm(dim=-1).mean()))
    rmse = errs[-1]
    assert rmse <= 0.5, f"2-anchor RMSE {rmse:.3f} > 0.5"
    ekf2 = RelativePositioningEKF(n)
    ekf2.reset(pos + 0.3 * torch.randn(n, 2), vel)
    for _ in range(40):
        out = ekf2.step(pos, vel, n_anchors=3, snr_db=20.0)
    rmse3 = float((out["pos_hat"] - pos).norm(dim=-1).mean())
    assert rmse3 <= 0.3, f"3-anchor RMSE {rmse3:.3f} > 0.3"
    u = fuse_uncertainty(torch.tensor([0.2, 0.8]), ekf.position_uncertainty()[:2])
    assert u.shape[0] == 2
    print(f"relative_positioning: OK (rmse2={rmse:.3f}, rmse3={rmse3:.3f})")


if __name__ == "__main__":
    self_check()
