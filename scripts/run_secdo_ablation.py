"""SECDO-v2.0 Ablation A1–A3 (Day 3) — no new scenarios.

A1: Reactive (no ĉ)
A2: Π_{B(c_t)} only  → SECDOSolver(fixed_alpha=0)
A3: α≡1            → SECDOSolver(fixed_alpha=1.0)
Full: adaptive α    → SECDOSolver(fixed_alpha=None)

Protocol:
  - UAV fast_drift, seeds 0..4, horizon 96, batch 16
  - Crash recovery (synthetic), cum violation

Outputs:
  results/secdo_v2/ablation/results.json
  results/secdo_v2/ablation/summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT = "checkpoints/secdo_v2/online/last.pt"
if not Path(CKPT).is_file():
    CKPT = "checkpoints/secdo_v2/joint/best.pt"


def _pick_ckpt() -> str:
    for p in (
        "checkpoints/secdo_v2/online/last.pt",
        "checkpoints/secdo_v2/joint/best.pt",
        "checkpoints/secdo_v2/pretrain/best.pt",
    ):
        if Path(p).is_file():
            return p
    raise FileNotFoundError("no secdo_v2 checkpoint")


def run_uav_ablation(ckpt: str) -> list[dict]:
    from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig, UAVBandwidthEnv
    from secdo.baselines.reactive import ReactiveSolver
    from secdo.baselines.secdo_solver import SECDOSolver
    from secdo.utils.seed import set_seed

    variants = [
        ("full_secdo", "secdo", None),
        ("A1_reactive", "reactive", None),
        ("A2_no_anticipatory", "secdo", 0.0),
        ("A3_fixed_alpha1", "secdo", 1.0),
    ]
    seeds = [0, 1, 2, 3, 4]
    rows = []
    for seed in seeds:
        for name, kind, fa in variants:
            set_seed(seed)
            env = UAVBandwidthEnv(
                UAVBandwidthConfig(horizon=96, seed=seed, regime="fast", vel_scale=0.4),
                device=DEVICE,
            )
            if kind == "reactive":
                solver = ReactiveSolver(eta=0.25, device=DEVICE)
            else:
                solver = SECDOSolver(
                    eta=0.25, device=DEVICE, ckpt=ckpt, fixed_alpha=fa
                )
                solver.name = name
            result = solver.solve(env, batch=16)
            gap_series = result.series.get("gap", [])
            reg_t = float(np.sum(gap_series)) if gap_series else float(result.metrics["gap"]) * 96
            row = {
                "protocol": "uav_fast_drift",
                "variant": name,
                "seed": seed,
                "Reg_T": reg_t,
                "gap_mean": float(result.metrics["gap"]),
                "violation": float(result.metrics["violation"]),
                "delta": float(result.metrics.get("delta", 0.0)),
                "chi": float(result.metrics.get("rho", 0.0)),
                "v2_mean_alpha": float(result.metrics.get("v2_mean_alpha", float("nan"))),
                "v2_PI": float(result.metrics.get("v2_PI", float("nan"))),
            }
            rows.append(row)
            print(
                f"[UAV|{name}|seed={seed}] Reg_T={reg_t:.5f} viol={row['violation']:.5f} "
                f"alpha={row['v2_mean_alpha']:.3f}",
                flush=True,
            )
    return rows


def run_crash_ablation() -> list[dict]:
    from secdo.experiments.crash_recovery.run import run_protocol

    mapping = [
        ("full_secdo", "secdo"),
        ("A1_reactive", "reactive"),
        ("A3_fixed_alpha1", "fixed_alpha"),
    ]
    # A2 ≡ reactive projection on crash synth (no learned ĉ) — same as A1 protocol
    rows = []
    for variant, method in mapping:
        row = run_protocol(method)
        cum = row["cum_violation"]
        out = {
            "protocol": "crash_recovery",
            "variant": variant,
            "sum_violation": float(row["sum_violation"]),
            "cum_final": float(cum[-1]) if cum else float(row["sum_violation"]),
            "cum_window_rise": float(cum[55] - cum[40]) if cum and len(cum) > 55 else float("nan"),
            "mean_alpha": float(row.get("mean_alpha", float("nan"))),
            "PI": float(row.get("PI", float("nan"))),
            "cum_violation": cum,
        }
        rows.append(out)
        print(
            f"[Crash|{variant}] sum_viol={out['sum_violation']:.4f} "
            f"window_rise={out['cum_window_rise']:.4f} alpha={out['mean_alpha']:.3f}",
            flush=True,
        )
    # A2 alias
    a1 = next(r for r in rows if r["variant"] == "A1_reactive")
    rows.append(
        {
            **{k: v for k, v in a1.items() if k != "variant"},
            "variant": "A2_no_anticipatory",
            "note": "crash-synth has no learned ĉ; A2 operationally equals A1 (Π_B(c_t))",
        }
    )
    return rows


def summarize(uav_rows: list[dict], crash_rows: list[dict]) -> dict:
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for r in uav_rows:
        buckets[r["variant"]].append(r)
    uav_sum = {}
    for v, rs in buckets.items():
        uav_sum[v] = {
            "n_seeds": len(rs),
            "Reg_T_mean": float(np.mean([x["Reg_T"] for x in rs])),
            "Reg_T_std": float(np.std([x["Reg_T"] for x in rs])),
            "violation_mean": float(np.mean([x["violation"] for x in rs])),
            "violation_std": float(np.std([x["violation"] for x in rs])),
            "mean_alpha": float(
                np.nanmean([x["v2_mean_alpha"] for x in rs])
                if any(np.isfinite(x["v2_mean_alpha"]) for x in rs)
                else float("nan")
            ),
        }
    crash_sum = {
        r["variant"]: {
            "sum_violation": r["sum_violation"],
            "cum_window_rise": r["cum_window_rise"],
            "mean_alpha": r["mean_alpha"],
        }
        for r in crash_rows
        if "cum_violation" in r or r.get("variant")
    }
    # strip series from crash_sum source already
    crash_sum = {
        r["variant"]: {
            "sum_violation": r["sum_violation"],
            "cum_window_rise": r.get("cum_window_rise"),
            "mean_alpha": r.get("mean_alpha"),
            **({"note": r["note"]} if "note" in r else {}),
        }
        for r in crash_rows
    }
    return {"uav_fast_drift": uav_sum, "crash_recovery": crash_sum, "ckpt": _pick_ckpt(), "device": DEVICE}


def main() -> int:
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)
    ckpt = _pick_ckpt()
    print("ckpt:", ckpt, flush=True)
    out_dir = Path("results/secdo_v2/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    uav_rows = run_uav_ablation(ckpt)
    crash_rows = run_crash_ablation()
    # persist crash without huge duplication issues — keep cum for full/A3/A1 only
    crash_store = []
    for r in crash_rows:
        crash_store.append({k: v for k, v in r.items()})

    payload = {"uav": uav_rows, "crash": crash_store}
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize(uav_rows, crash_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote", out_dir / "summary.json", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
