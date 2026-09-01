#!/usr/bin/env python
"""Phase 2B–2D: SECDO UAV regimes + theory metrics + ablations.

Usage:
  E:\\ANACONDA\\envs\\dpg_hrl\\python.exe -u scripts/run_secdo_phase2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig
from experiments.secdo_uav.runner import RunConfig, run_once


def main():
    out_dir = ROOT / "paper" / "ac_dsgf_v2" / "experiments" / "phase2_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes = {
        "slow": UAVBandwidthConfig(regime="slow", horizon=96, seed=0, vel_scale=0.4),
        "fast": UAVBandwidthConfig(regime="fast", horizon=96, seed=1, vel_scale=0.4),
        "stress": UAVBandwidthConfig(
            regime="stress", horizon=96, seed=2, vel_scale=0.5, stress_eps=0.35, stress_delta=0.25
        ),
    }

    # Core methods on fast regime + cross-regime SECDO/reactive
    jobs = []
    for reg_name, ecfg in regimes.items():
        for method in ("secdo", "reactive", "oracle"):
            jobs.append((reg_name, method, ecfg))
    # Ablations on fast (A: no latent, B: no constraint dyn, C: reactive=Π_B)
    for method in ("no_latent", "no_constraint_dyn"):
        jobs.append(("fast", method, regimes["fast"]))

    results = []
    for reg_name, method, ecfg in jobs:
        print(f"=== regime={reg_name} method={method} ===", flush=True)
        want_series = reg_name == "fast" and method in ("secdo", "reactive", "oracle")
        rcfg = RunConfig(
            method=method,
            train_steps=400,
            batch=24,
            seed=0,
            device="cpu",
            include_series=want_series,
        )
        # copy env cfg with distinct seed per job
        ecfg = UAVBandwidthConfig(**{**ecfg.__dict__, "seed": hash((reg_name, method)) % 10000})
        summary = run_once(ecfg, rcfg)
        summary["regime"] = reg_name
        results.append(summary)
        print(
            f"  gap={summary['mean_gap']:.4f} viol={summary['mean_viol']:.4f} "
            f"delta={summary['mean_delta']:.4f} rho={summary['mean_rho']:.4f} "
            f"frac(δ<ρ)={summary['frac_delta_lt_rho']:.2f} "
            f"eps+δ={summary['mean_eps_plus_delta']:.4f}",
            flush=True,
        )

    path = out_dir / "phase2_summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote", path, flush=True)

    # Theory-prediction scatter points (SECDO across regimes)
    scatter = [
        {
            "regime": r["regime"],
            "method": r["method"],
            "x_eps_plus_delta": r["mean_eps_plus_delta"],
            "y_gap": r["mean_gap"],
        }
        for r in results
        if r["method"] == "secdo"
    ]
    (out_dir / "theory_prediction_scatter.json").write_text(
        json.dumps(scatter, indent=2), encoding="utf-8"
    )
    print("Theory plot points:", scatter, flush=True)

    # Quick narrative checks
    fast = {r["method"]: r for r in results if r["regime"] == "fast"}
    if "secdo" in fast and "reactive" in fast:
        print(
            "FAST: SECDO viol vs reactive:",
            fast["secdo"]["mean_viol"],
            fast["reactive"]["mean_viol"],
            flush=True,
        )
        print(
            "FAST: frac(δ<ρ) SECDO:",
            fast["secdo"]["frac_delta_lt_rho"],
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
