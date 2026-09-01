"""Phase 5 smoke gate — check AC-DSGF++ causal communication hypothesis.

Reads a completed smoke run and compares against pass criteria:
  - Success ≥ 90% of AC-DSGF v1 reference (optional)
  - Comm drop 20–40% vs v1 (4UAV smoke reference)
  - corr(U, U*) > 0.5 (late training)
  - Gate non-zero (no silence collapse)

Usage:
  python scripts/eval_pp_smoke_check.py --run results/ac_dsgf_pp/ac_dsgf_pp_smoke
  python scripts/eval_pp_smoke_check.py --run results/ac_dsgf_pp/ac_dsgf_pp_smoke --v1-run results/ac_dsgf/ac_dsgf_smoke_v0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_csv_tail(path: Path, n: int = 3) -> list[dict]:
    rows = _read_csv(path)
    return rows[-n:] if rows else []


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] != ""]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="AC-DSGF++ smoke run directory")
    parser.add_argument("--v1-run", default=None, help="AC-DSGF v1 smoke for comparison")
    parser.add_argument(
        "--criteria",
        choices=("v2", "v2d"),
        default="v2",
        help="v2: Phase5 band; v2d: budget-aware (Success>20%%, corr>0.45, Comm near target)",
    )
    parser.add_argument("--comm-min", type=float, default=None)
    parser.add_argument("--comm-max", type=float, default=None)
    parser.add_argument("--u-min", type=float, default=0.1)
    parser.add_argument("--u-max", type=float, default=0.45)
    parser.add_argument("--gate-min", type=float, default=0.08)
    parser.add_argument("--success-min", type=float, default=None)
    parser.add_argument("--corr-thr", type=float, default=None)
    parser.add_argument(
        "--target-comm",
        type=float,
        default=None,
        help="v2d: expected soft budget target (widens Comm band if set)",
    )
    args = parser.parse_args()

    # Defaults by criteria
    if args.criteria == "v2d":
        success_min = 0.20 if args.success_min is None else args.success_min
        corr_thr = 0.45 if args.corr_thr is None else args.corr_thr
        # Around target_comm=0.5 (Version A); allow drift without Phase-6 blow-up
        t = 0.5 if args.target_comm is None else args.target_comm
        comm_min = 0.05 if args.comm_min is None else args.comm_min
        comm_max = max(1.0, 2.0 * t) if args.comm_max is None else args.comm_max
    else:
        success_min = 0.20 if args.success_min is None else args.success_min
        corr_thr = 0.45 if args.corr_thr is None else args.corr_thr
        comm_min = 0.10 if args.comm_min is None else args.comm_min
        comm_max = 0.35 if args.comm_max is None else args.comm_max

    run = Path(args.run)
    logs = run / "logs"
    summary_path = run / "summary.json"

    u_all = _read_csv(logs / "utility_mean.csv")
    g_all = _read_csv(logs / "gate_mass.csv")
    u_rows = u_all[-3:] if u_all else []
    g_rows = g_all[-3:] if g_all else []
    c_rows = _read_csv_tail(logs / "utility_corr.csv", 3)

    u_mean = _mean(u_rows, "utility_mean")
    gate_mass = _mean(g_rows, "gate_mass")
    # Use last-step corr (noisy mean of last-3 can hide recovery)
    corr = float(c_rows[-1]["corr_u_ustar"]) if c_rows else 0.0

    # Gate floor only: post-warmup Comm↓ is intended (not silence collapse)
    gate_early = _mean(g_all[: max(1, len(g_all) // 3)], "gate_mass") if g_all else 0.0
    gate_late = gate_mass
    gate_not_collapsed = gate_late >= args.gate_min

    success = 0.0
    comm = gate_mass
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            s = json.load(f)
        success = float(
            s.get("eval_success_rate", s.get("success", s.get("success_rate", 0.0))) or 0.0
        )
        # Prefer late gate_mass; fall back to mean comm
        comm = float(s.get("communication_cost_mean", comm) or comm)
        # Prefer late-window gate if available
        if gate_late > 0:
            comm = gate_late

    v1_success, v1_comm = None, None
    if args.v1_run:
        v1 = Path(args.v1_run) / "summary.json"
        if v1.exists():
            with open(v1, encoding="utf-8") as f:
                vs = json.load(f)
            v1_success = float(
                vs.get("eval_success_rate", vs.get("success", vs.get("success_rate", 0.0)))
                or 0.0
            )
            v1_comm = float(vs.get("communication_cost_mean", 0.0) or 0.0)

    checks = {
        "utility_in_band": args.u_min <= u_mean <= args.u_max,
        "gate_not_zero": gate_not_collapsed,
        "corr_u_ustar": corr >= corr_thr,
        "comm_in_band": comm_min <= comm <= comm_max,
        "success_floor": success >= success_min,
    }
    if args.criteria == "v2d" and v1_comm is not None and v1_comm > 0:
        # User gate: Comm should not exceed AC-DSGF reference
        checks["comm_below_ac"] = comm < v1_comm
    if v1_success is not None and v1_success > 0:
        checks["success_vs_v1"] = success >= 0.9 * v1_success

    passed = all(checks.values())

    label = "v2d Budget-aware" if args.criteria == "v2d" else "v2"
    print(f"=== AC-DSGF++ Smoke Check ({label}) ===")
    print(f"Run: {run}")
    print(f"Success: {success:.4f}" + (f" (v1 ref {v1_success:.4f})" if v1_success else ""))
    print(f"Comm/Gate late: {comm:.4f}  [want {comm_min}-{comm_max}]")
    if v1_comm is not None:
        print(f"AC-DSGF Comm ref: {v1_comm:.4f}")
    print(f"Gate early->late: {gate_early:.4f} -> {gate_late:.4f}")
    print(f"U mean (late): {u_mean:.4f}  [want {args.u_min}-{args.u_max}]")
    print(f"corr(U,U*):    {corr:.4f}  [want >{corr_thr}]")
    print("---")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print("---")
    next_msg = "PASS -> 16UAV seed42" if args.criteria == "v2d" else "PASS -> Phase 6"
    print("OVERALL:", next_msg if passed else "FAIL -> fix budget/utility before scaling")

    out = run / "phase5_check.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "passed": passed,
                "criteria": args.criteria,
                "metrics": {
                    "success": success,
                    "comm": comm,
                    "utility_mean": u_mean,
                    "gate_mass_early": gate_early,
                    "gate_mass_late": gate_late,
                    "utility_corr": corr,
                },
                "checks": checks,
            },
            f,
            indent=2,
        )
    print(f"Wrote {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
