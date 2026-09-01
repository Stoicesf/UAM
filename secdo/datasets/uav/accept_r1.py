"""R1 acceptance: new teacher matches legacy call path on fixed seeds.

Usage:
  python -m secdo.datasets.uav.accept_r1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.datasets.uav.channel import ChannelConfig
from secdo.datasets.uav.generator import GenerateConfig, generate_trajectory
from secdo.datasets.uav.mobility import step_positions
from secdo.datasets.uav.teacher import capacity_teacher, pack_state

from secdo.datasets.uav.teacher import capacity_teacher, pack_state

# Legacy alias path (same implementation)
legacy = __import__("secdo.datasets.uav.teacher", fromlist=["system_capacity_teacher"])


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = ChannelConfig(bandwidth_hz=1.0, comm_radius=2.5)

    pos = (torch.rand(16, 8, 2, device=device) - 0.5) * 4.0
    vel = (torch.rand(16, 8, 2, device=device) - 0.5) * 0.8

    c_new = capacity_teacher(pos, cfg)
    c_old = legacy.system_capacity_teacher(pos, cfg)
    pack_new = pack_state(pos, vel, cfg=cfg)
    pack_old = legacy.build_state_features(pos, vel, cfg=cfg)

    ok_c = torch.allclose(c_new, c_old, atol=1e-6, rtol=1e-6)
    ok_pack = torch.allclose(pack_new["c_teacher"], pack_old["c_teacher"], atol=1e-6, rtol=1e-6)
    ok_feat = torch.allclose(pack_new["feat"], pack_old["feat"], atol=1e-6, rtol=1e-6)

    # kinematics alias
    p2a, v2a = step_positions(pos.clone(), vel.clone(), dt=0.15, box=3.0)
    torch.manual_seed(1)
    p2b, v2b = legacy.step_positions(pos.clone(), vel.clone(), dt=0.15, box=3.0)
    # step has RNG — compare under same seed
    torch.manual_seed(123)
    p2a, v2a = step_positions(pos.clone(), vel.clone(), dt=0.15, box=3.0)
    torch.manual_seed(123)
    p2b, v2b = legacy.step_positions(pos.clone(), vel.clone(), dt=0.15, box=3.0)
    ok_step = torch.allclose(p2a, p2b) and torch.allclose(v2a, v2b)

    # generator internal teacher check + tiny offline dump
    gcfg = GenerateConfig(trajectories=2, length=16, seed=0, regime="fast_drift")
    tr = generate_trajectory(gcfg, 0, channel=cfg)
    assert tr["c"].shape[0] == 16

    print("R1 acceptance", flush=True)
    print(f"  teacher new vs legacy: {ok_c}  max|Δ|={float((c_new-c_old).abs().max()):.2e}", flush=True)
    print(f"  pack c_teacher:        {ok_pack}", flush=True)
    print(f"  pack feat:             {ok_feat}", flush=True)
    print(f"  step_positions:        {ok_step}", flush=True)
    print(f"  sample c mean:         {float(tr['c'].mean()):.6f}", flush=True)

    if not (ok_c and ok_pack and ok_feat and ok_step):
        print("FAIL", flush=True)
        return 1
    print("PASS accept_r1", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
