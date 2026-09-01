"""Exp.3 — Predictability boundary: PI vs violation reduction.

Usage:
  python -m secdo.experiments.pi_boundary.run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.experiments.synthetic_convex.env import SynthConfig, SyntheticConvexEnv
from secdo.experiments.synthetic_convex.run import _run_method


def main() -> int:
    out = Path("results/secdo/pi_boundary")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sigma in (0.01, 0.05, 0.1, 0.2, 0.5):
        cfg = SynthConfig(horizon=100, drift_amp=0.25, seed=1)
        env_a = SyntheticConvexEnv(cfg)
        env_r = SyntheticConvexEnv(cfg)
        secdo = _run_method(env_a, "secdo", pred_noise=sigma)
        reactive = _run_method(env_r, "reactive", pred_noise=sigma)
        reduction = reactive["sum_violation"] - secdo["sum_violation"]
        row = {
            "sigma": sigma,
            "PI": secdo["PI"],
            "frac_PI_lt_1": secdo["frac_PI_lt_1"],
            "violation_secdo": secdo["sum_violation"],
            "violation_reactive": reactive["sum_violation"],
            "violation_reduction": reduction,
        }
        rows.append(row)
        print(row, flush=True)

    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    xs = [r["PI"] for r in rows]
    ys = [r["violation_reduction"] for r in rows]
    ax.plot(xs, ys, "o-", color="#c0392b")
    ax.axvline(1.0, color="k", ls="--", label="PI=1")
    ax.set_xlabel("PI = δ/χ")
    ax.set_ylabel("Violation reduction (reactive − SECDO)")
    ax.set_title("Predictability boundary")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_pi_boundary.pdf")
    fig.savefig(out / "fig_pi_boundary.png", dpi=160)
    plt.close(fig)
    print("Wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
