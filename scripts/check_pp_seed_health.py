"""Phase 6 mid/post-seed health check — Success / Comm / corr / Precision.

Priority: Success hold > corr(U,U*) > Comm reduction.
Does NOT modify training. Run after a seed finishes (or mid-run on logs).

Usage:
  python scripts/check_pp_seed_health.py --run results/ac_dsgf_pp/uav16/s42
  python scripts/check_pp_seed_health.py --run results/ac_dsgf_pp/uav16/s42 --mid
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Frozen AC-DSGF reference (Table I)
AC_V1_SUCCESS = 0.0395
AC_V1_COMM = 0.4327
AC_V1_CEI = 0.0913


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] != ""]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument(
        "--mid",
        action="store_true",
        help="Mid-run: use logs only (no summary required)",
    )
    parser.add_argument("--corr-thr", type=float, default=0.5)
    parser.add_argument("--comm-target", type=float, default=0.25)
    parser.add_argument("--success-floor", type=float, default=0.035)
    args = parser.parse_args()

    run = Path(args.run)
    logs = run / "logs"
    summary_path = run / "summary.json"

    corr_rows = _read_csv(logs / "utility_corr.csv")
    gate_rows = _read_csv(logs / "gate_mass.csv")
    prec_rows = _read_csv(logs / "comm_precision.csv")
    if not prec_rows and corr_rows:
        # precision may be column in utility_corr
        prec_rows = corr_rows

    late_n = min(5, len(corr_rows)) if corr_rows else 0
    corr_late = _mean(corr_rows[-late_n:], "corr_u_ustar") if late_n else float("nan")
    corr_last = float(corr_rows[-1]["corr_u_ustar"]) if corr_rows else float("nan")
    gate_late = _mean(gate_rows[-late_n:], "gate_mass") if gate_rows else float("nan")
    gate_early = _mean(gate_rows[: max(1, len(gate_rows) // 5)], "gate_mass") if gate_rows else float("nan")
    prec_late = _mean(prec_rows[-late_n:], "comm_precision") if prec_rows else float("nan")

    success = float("nan")
    comm = gate_late
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        paper = s.get("paper_metrics", s)
        success = float(paper.get("success", s.get("success", 0.0)))
        if success > 1.0:
            success = success / 100.0
        comm = float(s.get("communication_cost_mean", paper.get("communication_cost", gate_late)) or gate_late)

    cei = success / (comm + 1e-6) if success == success else float("nan")

    # Case classification (single-seed peek; 5-seed overrides later)
    case = "PENDING"
    notes = []
    if args.mid or not summary_path.exists():
        case = "MID-RUN"
        if corr_late == corr_late and corr_late >= args.corr_thr:
            notes.append("corr OK so far")
        elif corr_late == corr_late:
            notes.append(f"corr {corr_late:.2f} < {args.corr_thr} (watch)")
        if gate_late == gate_late and gate_early == gate_early:
            if gate_late < 0.05 and gate_early > 1.0:
                notes.append("WARNING: possible silence collapse")
            elif gate_late < gate_early * 0.5:
                notes.append("Comm compressing (expected after warm-up)")
            else:
                notes.append("Gate still open (warm-up phase OK)")
    else:
        # Priority: Success > corr > Comm
        succ_ok = success >= args.success_floor
        corr_ok = corr_late >= args.corr_thr or corr_last >= args.corr_thr
        comm_ok = comm <= args.comm_target
        over_compress = comm < 0.05 and success < 0.9 * AC_V1_SUCCESS

        if over_compress or (not corr_ok and not succ_ok):
            case = "C"
            notes.append("over-compression or utility not learned")
        elif succ_ok and corr_ok and comm < 0.20:
            case = "A"
            notes.append("Success hold + strong Comm↓ + corr")
        elif succ_ok and corr_ok and (comm_ok or comm < AC_V1_COMM * 0.5):
            case = "B"
            notes.append("comparable Success + Comm↓ ≥50% + corr")
        elif succ_ok and not comm_ok:
            case = "B-weak"
            notes.append("Success OK but Comm not yet <0.25 — check 5-seed mean")
        else:
            case = "C"
            notes.append("needs λ reweight (no structure change)")

    print("=== AC-DSGF++ Seed Health Check ===")
    print(f"Run: {run}")
    print(f"Mode: {'mid-run logs' if args.mid or not summary_path.exists() else 'final summary'}")
    print(f"Steps logged: {len(gate_rows)}")
    print("---")
    print(f"Success:        {success*100:.2f}%" if success == success else "Success:        (pending)")
    print(f"Comm / Gate:    {comm:.4f}   [v1 ref {AC_V1_COMM:.2f}; target <{args.comm_target}]")
    print(f"Gate early→late:{gate_early:.3f} → {gate_late:.3f}")
    print(f"corr(U,U*) late:{corr_late:.3f}  last={corr_last:.3f}  [want >{args.corr_thr}]")
    print(f"Comm Precision: {prec_late:.3f}")
    if cei == cei:
        print(f"CEI:            {cei:.4f}   [v1 ref {AC_V1_CEI:.3f}]")
    print("---")
    print(f"CASE: {case}")
    for n in notes:
        print(f"  · {n}")
    print("---")
    print("Priority reminder: Success hold > Utility corr > Comm reduction")

    out = run / "seed_health.json"
    payload = {
        "case": case,
        "success": success if success == success else None,
        "comm": comm if comm == comm else None,
        "corr_late": corr_late if corr_late == corr_late else None,
        "corr_last": corr_last if corr_last == corr_last else None,
        "comm_precision": prec_late if prec_late == prec_late else None,
        "cei": cei if cei == cei else None,
        "gate_early": gate_early if gate_early == gate_early else None,
        "gate_late": gate_late if gate_late == gate_late else None,
        "notes": notes,
        "ref_ac_dsgf": {"success": AC_V1_SUCCESS, "comm": AC_V1_COMM, "cei": AC_V1_CEI},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
