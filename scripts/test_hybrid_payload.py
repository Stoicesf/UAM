#!/usr/bin/env python3
"""HybridPayloadDynamics smoke: taut → slack → taut, no NaN."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environments.dynamics.hybrid_payload import HybridPayloadDynamics


def _ring(n: int, r: float) -> torch.Tensor:
    ang = torch.linspace(0, 2 * 3.14159, n + 1)[:-1]
    return torch.stack([r * torch.cos(ang), r * torch.sin(ang)], dim=-1)


def main() -> int:
    n = 4
    L0 = 2.0
    pd = HybridPayloadDynamics(L0=L0, hysteresis=0.05, spring_k=50.0, damping_d=5.0)
    payload_pos = torch.zeros(2)
    payload_vel = torch.zeros(2)
    uav_vel = torch.zeros(n, 2)

    # 1) taut
    out1 = pd(_ring(n, L0 + 0.2), uav_vel, payload_pos, payload_vel, dt=0.05)
    assert bool((out1["state"] == 1).all()), f"expected taut, got {out1['state']}"
    assert out1["forces"].shape == (n, 2)
    assert not bool(torch.isnan(out1["new_payload_pos"]).any())

    # 2) slack
    out2 = pd(
        _ring(n, L0 - 0.3),
        uav_vel,
        out1["new_payload_pos"],
        out1["new_payload_vel"],
        dt=0.05,
    )
    assert bool((out2["state"] == 0).all()), f"expected slack, got {out2['state']}"
    assert bool((out2["tensions"] == 0).all())
    assert not bool(torch.isnan(out2["new_payload_vel"]).any())

    # 3) re-taut (with relative motion for impact path)
    uav_pos = _ring(n, L0 + 0.3)
    uav_vel_in = -0.3 * (uav_pos / torch.norm(uav_pos, dim=-1, keepdim=True))
    out3 = pd(
        uav_pos,
        uav_vel_in,
        out2["new_payload_pos"],
        out2["new_payload_vel"],
        dt=0.05,
    )
    assert bool((out3["state"] == 1).all()), f"expected taut again, got {out3['state']}"
    assert not bool(torch.isnan(out3["new_payload_pos"]).any())
    assert not bool(torch.isnan(out3["tensions"]).any())

    print("test_hybrid_payload: 3/3 OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
