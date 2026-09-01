"""Exp.1 — Synthetic convex theory validation (regret + bound proxy).

Usage:
  python -m secdo.experiments.synthetic_convex.run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.evaluation.regret import RegretTracker
from secdo.evaluation.violation import ViolationTracker
from secdo.experiments.synthetic_convex.env import SynthConfig, SyntheticConvexEnv
from secdo.optimizer.projected_gradient import gradient_step
from secdo.optimizer.reactive_projection import reactive_project
from secdo.optimizer.secdo_optimizer import SECDOOptimizer


def _peek_next_c(env: SyntheticConvexEnv) -> torch.Tensor:
    c0, b0, t0 = env.c.clone(), env.b.clone(), env.t
    obs_n = env.step()
    c_next = obs_n["c_teacher"].clone()
    env.c, env.b, env.t = c0, b0, t0
    return c_next


def _run_method(env: SyntheticConvexEnv, method: str, eta: float = 0.2, pred_noise: float = 0.05):
    obs = env.reset(batch=32)
    pref = obs["pref"]
    x = reactive_project(pref.clone(), obs["c_teacher"])
    opt = SECDOOptimizer(eta=eta)
    regret = RegretTracker()
    viol = ViolationTracker()
    c_prev = None
    while not env.done:
        c_t = obs["c_teacher"]
        c_true_next = _peek_next_c(env)
        if method == "oracle":
            c_hat = c_true_next
        elif method == "reactive":
            c_hat = c_t
        else:
            c_hat = c_true_next + pred_noise * torch.randn_like(c_true_next)

        # True drift χ for PI (theory); online systems can use |c_t-c_{t-1}|
        chi = (c_true_next - c_t).abs().clamp(min=1e-2)
        delta_hat = (c_hat - c_true_next).abs()
        if method == "reactive":
            x = reactive_project(gradient_step(x, pref, eta), c_t)
            alpha = torch.zeros_like(c_t)
            pi = delta_hat / chi
        elif method == "oracle":
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
        regret.update(x, obs["x_star"], pref)
        delta = float(delta_hat.mean())
        chi_v = float(chi.mean())
        viol.update(
            x, c_next, delta=delta, chi=chi_v, PI=float(pi.mean()), alpha=float(alpha.mean())
        )
        c_prev = c_t
    return {**regret.summary(), **viol.summary(), "method": method}


def main() -> int:
    out = Path("results/secdo/synthetic_convex")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for noise in (0.01, 0.05, 0.1, 0.2):
        for method in ("secdo", "reactive", "oracle"):
            cfg = SynthConfig(horizon=120, drift_amp=0.2, noise_sigma=noise, seed=0)
            env = SyntheticConvexEnv(cfg)
            row = _run_method(env, method, pred_noise=noise)
            row["noise"] = noise
            rows.append(row)
            print(
                f"noise={noise} {method}: Reg={row['Reg_T']:.4f} viol={row['violation']:.4f} "
                f"PI={row['PI']:.3f} frac(PI<1)={row['frac_PI_lt_1']:.2f}",
                flush=True,
            )
    (out / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("Wrote", out / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
