# -*- coding: utf-8 -*-
"""Gate 2 — Twin evaluation hook validation (NO training).

Sanity cases (shared state only):
  Case 1: G_t = G*     → ε_G ≈ 0, Δa ≈ 0
  Case 2: G_t = empty  → ε_G > 0 (and usually Δa > 0)
  Case 3: K↑           → ε_G generally ↓ (sanity trend, not paper fig)

Usage:
  conda run -n dpg_hrl python scripts/validate_tro_twin_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ac_dsgf import ACDSGF
from models.residual_policy import ResidualGuidanceActor
from tro.twin.twin_evaluator import TwinEvaluator


def _make_obs(N: int, seed: int, spread: float = 0.25) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(1, N, 64, generator=g)
    obs[..., :2] = torch.randn(1, N, 2, generator=g) * spread
    obs[..., 2:4] = torch.randn(1, N, 2, generator=g) * 0.05
    return obs


def build_stack(N: int, seed: int = 0):
    torch.manual_seed(seed)
    enc = ACDSGF(obs_dim=64, hidden_dim=64, guidance_dim=6, comm_radius=1.5)
    enc.eval()
    actor = ResidualGuidanceActor(
        obs_dim=64,
        phi_dim=6,
        action_dim=2,
        n_agents=N,
        hidden_dim=64,
        device="cpu",
        beta_init=1.0,
    )
    actor.eval()
    return enc, actor


def case1_identity(N: int = 16) -> dict:
    enc, actor = build_stack(N)
    ev = TwinEvaluator(enc, actor, run_prefix="tro_gate2_case1")
    obs = _make_obs(N, seed=11)
    r = ev.step(obs, t=0, sparse_mode="full", log=True)
    # numerical tolerance
    ok = r.epsilon_G < 1e-5 and (r.delta_a is not None and r.delta_a < 1e-5)
    summary = {
        "case": 1,
        "name": "full_equals_full",
        "epsilon_G": r.epsilon_G,
        "delta_a": r.delta_a,
        "pass": ok,
        "run_dir": str(ev.run_dir),
        "state_id": r.state_id,
    }
    ev.write_summary(summary)
    return summary


def case2_empty(N: int = 16) -> dict:
    enc, actor = build_stack(N, seed=1)
    ev = TwinEvaluator(enc, actor, run_prefix="tro_gate2_case2")
    obs = _make_obs(N, seed=22)
    r = ev.step(obs, t=0, sparse_mode="empty", log=True)
    ok = r.epsilon_G > 1e-4
    # Δa usually >0 under residual β>0; soft warn if not
    summary = {
        "case": 2,
        "name": "empty_graph",
        "epsilon_G": r.epsilon_G,
        "delta_a": r.delta_a,
        "pass": ok,
        "delta_a_positive": bool(r.delta_a is not None and r.delta_a > 1e-6),
        "run_dir": str(ev.run_dir),
        "state_id": r.state_id,
    }
    ev.write_summary(summary)
    return summary


def case3_k_monotone(N: int = 16, Ks=(2, 4, 8), trials: int = 8) -> dict:
    """Sanity: larger K → smaller mean ε_G (not a paper claim)."""
    enc, actor = build_stack(N, seed=2)
    ev = TwinEvaluator(enc, actor, run_prefix="tro_gate2_case3")
    means = {}
    for K in Ks:
        eps = []
        das = []
        for i in range(trials):
            obs = _make_obs(N, seed=1000 + K * 10 + i)
            r = ev.step(obs, t=K * 100 + i, sparse_mode="fixed_k", k_fixed=K, log=True)
            eps.append(r.epsilon_G)
            if r.delta_a is not None:
                das.append(r.delta_a)
        means[K] = {
            "mean_epsilon_G": float(sum(eps) / len(eps)),
            "mean_delta_a": float(sum(das) / len(das)) if das else None,
        }
    # Check non-increasing trend K=2 → 4 → 8 (allow tiny noise)
    e2, e4, e8 = means[2]["mean_epsilon_G"], means[4]["mean_epsilon_G"], means[8]["mean_epsilon_G"]
    # Soft: e2 >= e4 * 0.95 and e4 >= e8 * 0.95  OR  e2 > e8
    ok = (e2 + 1e-6 >= e8) and (e2 + 1e-6 >= e4 * 0.9)
    summary = {
        "case": 3,
        "name": "K_increase_epsilon_decrease",
        "by_K": means,
        "pass": ok,
        "note": "sanity trend only; not Lemma 2 proof",
        "run_dir": str(ev.run_dir),
    }
    ev.write_summary(summary)
    return summary


def recoverability_check(run_dir: Path) -> bool:
    """Gate 2 requirement: ε_G and Δa recoverable from logs."""
    csv = run_dir / "twin" / "epsilon_delta.csv"
    if not csv.exists():
        return False
    lines = csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return False
    # message files exist for last logged t
    mf = list((run_dir / "twin" / "message_full").glob("*.npy"))
    ms = list((run_dir / "twin" / "message_sparse").glob("*.npy"))
    return bool(mf) and bool(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16)
    args = ap.parse_args()

    results = []
    print("=== Gate 2A/B: twin sanity cases ===")
    c1 = case1_identity(args.N)
    results.append(c1)
    print(
        f"  Case1 identity: ε={c1['epsilon_G']:.3e} Δa={c1['delta_a']:.3e} "
        f"[{'PASS' if c1['pass'] else 'FAIL'}]"
    )

    c2 = case2_empty(args.N)
    results.append(c2)
    print(
        f"  Case2 empty:    ε={c2['epsilon_G']:.3e} Δa={c2['delta_a']:.3e} "
        f"[{'PASS' if c2['pass'] else 'FAIL'}] "
        f"(Δa>0={c2['delta_a_positive']})"
    )

    c3 = case3_k_monotone(args.N)
    results.append(c3)
    print(f"  Case3 K trend:  {c3['by_K']} [{'PASS' if c3['pass'] else 'FAIL'}]")

    recover_ok = all(recoverability_check(Path(r["run_dir"])) for r in results)
    print(f"  Log recoverability: [{'PASS' if recover_ok else 'FAIL'}]")

    all_pass = all(r["pass"] for r in results) and recover_ok
    out = ROOT / "paper" / "ac_dsgf_tro" / "experiments" / "_gate2_last_result.json"
    out.write_text(
        json.dumps({"all_pass": all_pass, "recoverability": recover_ok, "cases": results}, indent=2),
        encoding="utf-8",
    )
    print("=== Gate 2 summary:", "PASS" if all_pass else "FAIL", "===")
    print("Wrote", out)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
