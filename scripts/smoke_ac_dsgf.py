"""Phase-0 AC-DSGF smoke: train + compare soft edge count vs DSGF v2 20k.

Usage:
  python scripts/smoke_ac_dsgf.py
  python scripts/smoke_ac_dsgf.py --train-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CFG = ROOT / "configs" / "ac_dsgf" / "ac_dsgf_smoke_v0.yaml"
DSGF_REF = ROOT / "results" / "dsgf" / "dsfg_v2_102k" / "summary.json"


def _mean_edges_from_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "communication_cost_mean" in data:
        return float(data["communication_cost_mean"])
    paper = data.get("paper_metrics", {})
    if "communication_cost" in paper:
        return float(paper["communication_cost"])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--run-name", default="ac_dsgf_smoke_v0")
    args = parser.parse_args()

    out_summary = ROOT / "results" / "ac_dsgf" / args.run_name / "summary.json"

    if not args.skip_train:
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--exp", str(SMOKE_CFG),
            "--run-name", args.run_name,
        ]
        print(">>>", " ".join(cmd))
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            sys.exit(rc)

    if args.train_only:
        return

    ac_edges = _mean_edges_from_summary(out_summary)
    dsgf_edges = _mean_edges_from_summary(DSGF_REF)

    report = {
        "ac_dsgf_summary": str(out_summary),
        "ac_dsgf_soft_edges": ac_edges,
        "dsgf_v2_ref_summary": str(DSGF_REF),
        "dsgf_v2_edges": dsgf_edges,
    }
    if ac_edges is not None and dsgf_edges is not None and dsgf_edges > 0:
        drop = (dsgf_edges - ac_edges) / dsgf_edges
        report["edge_drop_ratio"] = round(drop, 4)
        report["pass_edges_ge_30pct"] = bool(drop >= 0.30)
        print(f"DSGF edges≈{dsgf_edges:.2f}  AC-DSGF edges≈{ac_edges:.2f}  drop={drop:.1%}")
        print("PASS" if report["pass_edges_ge_30pct"] else "NEED TUNING (target drop≥30%)")
    else:
        print("[warn] missing edge stats for comparison")
        if out_summary.exists():
            data = json.loads(out_summary.read_text(encoding="utf-8"))
            print("AC success:", data.get("success") or data.get("paper_metrics", {}).get("success"))
            print("AC communication_cost_mean:", data.get("communication_cost_mean"))

    out = ROOT / "results" / "ac_dsgf" / args.run_name / "smoke_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
