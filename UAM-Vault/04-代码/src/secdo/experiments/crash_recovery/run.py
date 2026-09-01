"""Exp.5 — Prediction crash recovery (Cor.4).

Usage:
  python -m secdo.experiments.crash_recovery.run
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

from secdo.evaluation.violation import ViolationTracker
from secdo.experiments.synthetic_convex.env import SynthConfig, SyntheticConvexEnv
from secdo.experiments.synthetic_convex.run import _peek_next_c
from secdo.optimizer.projected_gradient import gradient_step
from secdo.optimizer.reactive_projection import reactive_project
from secdo.optimizer.secdo_optimizer import SECDOOptimizer


def run_protocol(method: str, T1: int = 40, T2: int = 55, T: int = 100, noise: float = 0.05):
    cfg = SynthConfig(horizon=T, drift_amp=0.2, seed=2)
    env = SyntheticConvexEnv(cfg)
    obs = env.reset(batch=32)
    pref = obs["pref"]
    x = reactive_project(pref.clone(), obs["c_teacher"])
    opt = SECDOOptimizer(eta=0.2)
    vtr = ViolationTracker()
    c_prev = None
    cum = []
    while not env.done:
        c_t = obs["c_teacher"]
        c_true_next = _peek_next_c(env)
        c_hat = c_true_next + noise * torch.randn_like(c_true_next)
        if T1 <= env.t < T2:
            c_hat = c_hat * 4.0

        chi = (c_true_next - c_t).abs().clamp(min=1e-2)
        delta_hat = (c_hat - c_true_next).abs()
        if method == "reactive":
            x = reactive_project(gradient_step(x, pref, 0.2), c_t)
            alpha = torch.zeros_like(c_t)
            pi = delta_hat / chi
        elif method == "fixed_alpha":
            x, alpha, pi = opt.step(
                x, None, c_hat, c_t, delta_hat=delta_hat, chi_hat=chi, pref=pref, fixed_alpha=1.0
            )
        else:
            x, alpha, pi = opt.step(
                x, None, c_hat, c_t, delta_hat=delta_hat, chi_hat=chi, pref=pref
            )

        obs = env.step()
        c_next = obs["c_teacher"]
        pref = obs["pref"]
        delta = float((c_hat - c_next).abs().mean())
        chi_v = float((c_next - c_t).abs().mean())
        vtr.update(x, c_next, delta=delta, chi=chi_v, PI=float(pi.mean()), alpha=float(alpha.mean()))
        cum.append(sum(vtr.viol))
        c_prev = c_t
    return {"method": method, "cum_violation": cum, **vtr.summary()}


def main() -> int:
    out = Path("results/secdo/crash_recovery")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for method in ("secdo", "fixed_alpha", "reactive"):
        row = run_protocol(method)
        rows.append(row)  # keep cum_violation for Cor.4 figures
        ax.plot(row["cum_violation"], label=method, lw=1.6)
        print(method, {k: row[k] for k in ("sum_violation", "PI", "mean_alpha")}, flush=True)
    ax.axvspan(40, 55, color="#f5b7b1", alpha=0.4, label="corrupt")
    ax.set_xlabel("t")
    ax.set_ylabel("cumulative violation")
    ax.set_title("Prediction crash recovery (Cor.4)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "fig_crash_recovery.pdf")
    fig.savefig(out / "fig_crash_recovery.png", dpi=160)
    plt.close(fig)
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
