#!/usr/bin/env python3
"""Phase-3 CSCBF: single-cable aggressive slam with/without CSCBF projection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from environments.dynamics.hybrid_payload import HybridPayloadDynamics
from models.transport.safety.cscbf_shield import CSCBFShield


def _ring(n: int, r: float) -> torch.Tensor:
    ang = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    return torch.stack([r * torch.cos(ang), r * torch.sin(ang)], dim=-1)


def _run(use_cscbf: bool, n_cycles: int = 80, n: int = 4, L0: float = 2.0) -> dict:
    pd = HybridPayloadDynamics(L0=L0, hysteresis=0.05, spring_k=150.0, damping_d=0.5)
    pd.reset_state(n)
    payload_pos = torch.zeros(2)
    payload_vel = torch.zeros(2)
    uav = _ring(n, L0 + 0.25)
    shield = CSCBFShield(
        n, cable_length=L0, safety_margin=0.1, alpha=3.0, slack_close_max=0.15
    )
    peaks: list[float] = []
    dt = 0.05
    i = 0  # only agent 0 is aggressive

    for _ in range(n_cycles):
        for slam_in in (True, False):
            for _s in range(12):
                diff = uav - payload_pos.unsqueeze(0)
                dist = torch.norm(diff, dim=-1, keepdim=True).clamp(min=1e-6)
                qhat = diff / dist
                u_nom = torch.zeros(n, 2)
                u_nom[i] = (-4.0 if slam_in else 4.5) * qhat[i]
                # hold others gently on ring
                ideal = _ring(n, L0 + 0.2) + payload_pos.unsqueeze(0)
                u_nom[1:] = 1.5 * (ideal[1:] - uav[1:])

                state = pd.state if pd.state.numel() == n else torch.ones(n, dtype=torch.long)
                tensions = torch.zeros(n, 1)
                u = (
                    shield.apply(
                        u_nom, uav, payload_pos, payload_vel, tensions, state, dt
                    )
                    if use_cscbf
                    else u_nom
                )
                uav = uav + u * dt
                out = pd(uav, u, payload_pos, payload_vel, dt)
                payload_pos, payload_vel = out["new_payload_pos"], out["new_payload_vel"]
                peaks.append(float(torch.norm(payload_vel)))
                tensions = out["tensions"]

    return {"peak_vel": max(peaks), "p95": sorted(peaks)[int(0.95 * (len(peaks) - 1))]}


def main() -> int:
    base = _run(use_cscbf=False)
    cscbf = _run(use_cscbf=True)
    print(
        f"switch compare: baseline_peak={base['peak_vel']:.3f} p95={base['p95']:.3f} | "
        f"cscbf_peak={cscbf['peak_vel']:.3f} p95={cscbf['p95']:.3f}"
    )
    assert base["peak_vel"] >= 0.8, base
    reduction = 1.0 - cscbf["peak_vel"] / max(base["peak_vel"], 1e-6)
    assert reduction >= 0.45, (base, cscbf, reduction)

    out_dir = ROOT / "experiment_results" / "hybrid_formal"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(
        ["baseline", "CSCBF"],
        [base["peak_vel"], cscbf["peak_vel"]],
        color=["#888", "#2a6"],
    )
    ax.set_ylabel("peak ||v_payload|| (m/s)")
    ax.set_title(f"Single-cable slam (reduction={reduction*100:.0f}%)")
    fig.tight_layout()
    png = out_dir / "vmas_switch_compare.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"wrote {png}")
    print("test_cscbf_switch: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
