#!/usr/bin/env python3
"""Phase-3/4 smoke: min-snap traj smoothness + APF min separation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from environments.scenarios.cooperative_transport import CooperativeTransportEnv
from models.transport.control.hierarchical import TransportHierarchicalController
from models.transport.planning.minimum_snap import MinimumSnapTrajectory


def test_traj_smooth() -> None:
    wps = [[0.0, 0.0], [2.5, 2.5], [5.0, 5.0]]
    traj = MinimumSnapTrajectory(wps, total_time=10.0)
    traj.generate()
    accs = []
    for t in np.linspace(0, 10, 101):
        _p, _v, a = traj.compute(float(t))
        accs.append(np.linalg.norm(a))
    accs = np.asarray(accs)
    # piecewise endpoints may spike; interior should be finite and not huge
    assert np.isfinite(accs).all()
    # compare to bang linear (infinite snap) — max |a| of quintic is moderate
    assert float(accs.max()) < 5.0
    print(f"phase3 traj: max|a|={accs.max():.3f}")


def test_traj_controller() -> None:
    env = CooperativeTransportEnv(
        n_agents=16, seed=0, max_steps=120, use_hybrid=True, target_pos=(5.0, 5.0)
    )
    obs, _ = env.reset(seed=0)
    ctl = TransportHierarchicalController(env, use_rise=True, use_traj=True)
    dists = []
    for _ in range(120):
        obs, _, done, _, info = env.step(ctl.act(obs))
        dists.append(float(info["payload_distance"]))
        if done:
            break
    assert dists[-1] < 1.0, dists[-1]
    print(f"phase3 ctrl: steps={info['step']} payload_d={dists[-1]:.3f}")


def test_apf_separation() -> None:
    env = CooperativeTransportEnv(
        n_agents=16,
        seed=2,
        max_steps=80,
        use_hybrid=True,
        obstacles=[(2.5, 2.5), (3.0, 3.5)],
    )
    obs, _ = env.reset(seed=2)
    ctl = TransportHierarchicalController(env, use_rise=True, use_shield=True)
    min_sep = 1e9
    for _ in range(80):
        obs, _, done, _, info = env.step(ctl.act(obs))
        dmat = torch.cdist(env.pos, env.pos)
        eye = torch.eye(env.n_agents)
        sep = (dmat + eye * 1e6).min().item()
        min_sep = min(min_sep, sep)
        if done:
            break
    assert min_sep >= 0.45, min_sep  # env _separate uses 0.45; shield targets 0.5
    print(f"phase4 apf: min_sep={min_sep:.3f} payload_d={info['payload_distance']:.3f}")


def main() -> int:
    test_traj_smooth()
    test_traj_controller()
    test_apf_separation()
    print("test_transport_phase34: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
