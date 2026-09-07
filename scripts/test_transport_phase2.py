#!/usr/bin/env python3
"""Phase-2 acceptance: formation error, RISE vs wind, cable_state."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from environments.scenarios.cooperative_transport import CooperativeTransportEnv
from models.transport.control.hierarchical import TransportHierarchicalController


def _run(wind: float, use_rise: bool, steps: int = 100, seed: int = 0) -> dict:
    env = CooperativeTransportEnv(
        n_agents=16,
        seed=seed,
        max_steps=steps,
        use_hybrid=True,
        wind_force=wind,
        target_pos=(5.0, 5.0),
    )
    obs, info = env.reset(seed=seed)
    ctl = TransportHierarchicalController(env, use_rise=use_rise)
    form_errs = []
    cable_ok = True
    for _ in range(steps):
        act = ctl.act(obs)
        obs, _, done, _, info = env.step(act)
        form_errs.append(float(info["formation_error"]))
        if "cable_state" in info:
            st = info["cable_state"]
            if any(s not in (0, 1) for s in st):
                cable_ok = False
        if done:
            break
    return {
        "payload_distance": float(info["payload_distance"]),
        "formation_error": form_errs[-1],
        "formation_error_mean": sum(form_errs) / max(len(form_errs), 1),
        "steps": info["step"],
        "cable_ok": cable_ok,
        "cable_state": info.get("cable_state"),
    }


def main() -> int:
    # formation under theory+hybrid
    r0 = _run(wind=0.0, use_rise=True, steps=100)
    print(
        f"formation: form_err={r0['formation_error']:.3f} "
        f"payload_d={r0['payload_distance']:.3f} cable_ok={r0['cable_ok']}"
    )
    assert r0["formation_error"] < 0.3, r0
    assert r0["cable_ok"]

    # RISE vs no-RISE under wind
    with_rise = _run(wind=0.5, use_rise=True, steps=150, seed=1)
    no_rise = _run(wind=0.5, use_rise=False, steps=150, seed=1)
    print(
        f"wind RISE: d={with_rise['payload_distance']:.3f} "
        f"no_RISE d={no_rise['payload_distance']:.3f}"
    )
    assert with_rise["payload_distance"] < 0.5, with_rise
    assert no_rise["payload_distance"] > 1.0, no_rise
    print("test_transport_phase2: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
