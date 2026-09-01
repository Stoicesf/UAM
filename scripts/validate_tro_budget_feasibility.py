# -*- coding: utf-8 -*-
"""Gate 1 — Budget feasibility validation (NO training, NO reward tuning).

Validates Theorem 1 projection:
    G_t = Π_B(S_t)  ⇒  V_B = max(0, d_i - k_i) = 0

Usage:
  python scripts/validate_tro_budget_feasibility.py
  python scripts/validate_tro_budget_feasibility.py --with-encoder --N 16 --K 4

Exit code 0 iff all synthetic + optional encoder checks pass (max V_B == 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.communication.budget_layer import (
    apply_topk_budget,
    apply_topk_fixed_k,
    degree_budget_violation,
    hard_adjacency,
)
from utils.tro_run import append_topology_step, create_run_dir, summarize_v_b, write_config


def _random_mask(B: int, N: int, p: float = 0.4) -> torch.Tensor:
    m = (torch.rand(B, N, N) < p).float()
    eye = torch.eye(N).unsqueeze(0)
    return m * (1.0 - eye)


def check_fixed_k(N: int, K: int, trials: int = 32, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    max_vb = 0.0
    mean_deg = []
    rhos = []
    for _ in range(trials):
        mask = _random_mask(2, N, p=0.5)
        scores = torch.rand(2, N, N, generator=g)
        gated = apply_topk_fixed_k(scores, mask, K)
        A = hard_adjacency(gated)
        feas = degree_budget_violation(A, mask, k_fixed=K)
        max_vb = max(max_vb, feas["V_B"])
        mean_deg.append(feas["mean_degree"])
        rhos.append(feas["rho"])
    return {
        "N": N,
        "K": K,
        "max_V_B": max_vb,
        "mean_degree": float(sum(mean_deg) / len(mean_deg)),
        "mean_rho": float(sum(rhos) / len(rhos)),
        "pass": max_vb == 0.0,
    }


def check_ratio(N: int, ratio: float, trials: int = 32, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    max_vb = 0.0
    for _ in range(trials):
        mask = _random_mask(2, N, p=0.5)
        scores = torch.rand(2, N, N, generator=g)
        gated = apply_topk_budget(scores, mask, ratio)
        A = hard_adjacency(gated)
        feas = degree_budget_violation(A, mask, budget_ratio=ratio)
        max_vb = max(max_vb, feas["V_B"])
    return {"N": N, "ratio": ratio, "max_V_B": max_vb, "pass": max_vb == 0.0}


def check_encoder(N: int, K: int, steps: int = 16, seed: int = 42) -> dict:
    """Untrained ACDSGF forward — verifies diagnostics wiring, not task return."""
    from models.ac_dsgf import ACDSGF

    torch.manual_seed(seed)
    enc = ACDSGF(obs_dim=64, hidden_dim=64, guidance_dim=6, comm_radius=1.5)
    enc.eval()
    enc.fixed_k = K
    enc.ablation_mode = "full"
    max_vb = 0.0
    degrees = []
    rhos = []
    run_dir = create_run_dir(prefix=f"tro_gate1_enc_N{N}_K{K}")
    write_config(
        run_dir,
        {
            "gate": 1,
            "purpose": "budget_feasibility_encoder_dry_run",
            "N": N,
            "K": K,
            "training": False,
        },
    )
    with torch.no_grad():
        for t in range(steps):
            obs = torch.randn(1, N, 64)
            # pack positions into first 2 dims with moderate spread so radius has edges
            pos = torch.randn(1, N, 2) * 0.3
            obs[..., :2] = pos
            obs[..., 2:4] = torch.randn(1, N, 2) * 0.05
            _, _, diag = enc(obs, positions=pos, apply_budget=True)
            vb = float(diag["V_B"])
            max_vb = max(max_vb, vb)
            degrees.append(float(diag["mean_degree"]))
            rhos.append(float(diag["rho_t"]))
            append_topology_step(
                run_dir,
                t=t,
                A_t=diag["A_t"],
                S_t=diag.get("S_t"),
                B_t=float(diag["B_t"]),
                C_t=float(diag["C_t"]),
                V_B=vb,
                degrees=diag["degrees"],
                rho_t=float(diag["rho_t"]),
                mean_degree=float(diag["mean_degree"]),
                meta={"K": K, "N": N},
            )
    summary = summarize_v_b(run_dir)
    summary.update(
        {
            "N": N,
            "K": K,
            "max_V_B": max_vb,
            "mean_degree": float(sum(degrees) / len(degrees)),
            "mean_rho": float(sum(rhos) / len(rhos)),
            "pass": max_vb == 0.0,
            "run_dir": str(run_dir),
        }
    )
    (run_dir / "metrics" / "gate1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main():
    ap = argparse.ArgumentParser(description="T-RO Gate 1: V_B=0 budget feasibility")
    ap.add_argument("--with-encoder", action="store_true", help="also dry-run untrained ACDSGF")
    ap.add_argument("--N", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--K", type=int, nargs="+", default=[2, 4, 8])
    args = ap.parse_args()

    rows = []
    all_pass = True
    print("=== Gate 1: synthetic fixed-K projection ===")
    for N in args.N:
        for K in args.K:
            r = check_fixed_k(N, K)
            rows.append(r)
            all_pass &= r["pass"]
            status = "PASS" if r["pass"] else "FAIL"
            print(
                f"  [{status}] N={N} K={K}  max_V_B={r['max_V_B']:.4g}  "
                f"mean_d={r['mean_degree']:.3f}  rho={r['mean_rho']:.4f}"
            )

    print("=== Gate 1: synthetic budget_ratio projection ===")
    for N in args.N:
        for ratio in (0.25, 0.5, 0.75):
            r = check_ratio(N, ratio)
            all_pass &= r["pass"]
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  [{status}] N={N} ratio={ratio}  max_V_B={r['max_V_B']:.4g}")

    enc_rows = []
    if args.with_encoder:
        print("=== Gate 1: untrained ACDSGF encoder dry-run ===")
        for N in args.N:
            for K in args.K:
                r = check_encoder(N, K)
                enc_rows.append(r)
                all_pass &= r["pass"]
                status = "PASS" if r["pass"] else "FAIL"
                print(
                    f"  [{status}] N={N} K={K}  max_V_B={r['max_V_B']:.4g}  "
                    f"mean_d={r['mean_degree']:.3f}  run={r['run_dir']}"
                )

    out = ROOT / "paper" / "ac_dsgf_tro" / "experiments" / "_gate1_last_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"all_pass": all_pass, "fixed_k": rows, "encoder": enc_rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== Gate 1 summary:", "PASS" if all_pass else "FAIL", "===")
    print("Wrote", out)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
