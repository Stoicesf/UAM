"""Post-processing after 5-seed baseline16 completes (P0 evidence chain).

Steps:
  1. Verify 20 summaries
  2. Stats: mean±std, 95% CI, AULC, t-test / Wilcoxon
  3. Figures: learning curves, AULC bar, success bar
  4. Export paper tables (CSV + JSON)
  5. Write pipeline status + next-task checklist

Usage:
  python scripts/pipeline_after_5seed.py
  python scripts/pipeline_after_5seed.py --force   # run even if <20 summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 3407, 2026, 1234, 8888]
METHODS = ["mappo", "gat", "transformer", "dsgf"]
METHOD_LABELS = {
    "mappo": "MAPPO",
    "gat": "GAT",
    "transformer": "Transformer",
    "dsgf": "DSGF (Ours)",
}


def count_summaries() -> tuple[int, list[str]]:
    missing = []
    n = 0
    for method in METHODS:
        for seed in SEEDS:
            p = ROOT / "results" / "baseline16_seeds" / method / f"s{seed}" / "summary.json"
            if seed == 42 and not p.exists():
                legacy = ROOT / "results" / "baseline16" / method / "summary.json"
                if legacy.exists():
                    n += 1
                    continue
            if p.exists():
                n += 1
            else:
                missing.append(f"{method}/s{seed}")
    return n, missing


def run_script(name: str, *args: str) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / name), *args]
    print(f"\n[pipeline] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(ROOT))


def export_5seed_csv(report_path: Path, out_csv: Path):
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    rows = []
    for row in report.get("methods", []):
        if row.get("n_seeds", 0) == 0:
            continue
        s, c, r, a = row["success"], row["collision"], row["reward"], row["aulc"]
        rows.append({
            "Method": METHOD_LABELS.get(row["method"], row["method"]),
            "Success_mean": s["mean"],
            "Success_std": s["std"],
            "Success_ci95_lo": s["ci95_lo"],
            "Success_ci95_hi": s["ci95_hi"],
            "Collision_mean": c["mean"],
            "Collision_std": c["std"],
            "Reward_mean": r["mean"],
            "Reward_std": r["std"],
            "AULC_mean": a["mean"],
            "AULC_std": a["std"],
            "n_seeds": row["n_seeds"],
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_status(report_path: Path, n_summaries: int, steps_done: list[str]):
    sig = {}
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            sig = json.load(f).get("significance", {})

    status = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "summaries_complete": n_summaries,
        "summaries_target": len(SEEDS) * len(METHODS),
        "p0_completed": steps_done,
        "p0_done": n_summaries >= len(SEEDS) * len(METHODS),
        "significance": sig,
        "next_tasks": [
            {
                "priority": "P1",
                "task": "Generalization (zero-shot obstacle / comm radius)",
                "status": "pending",
                "note": "Do not retrain; eval from frozen checkpoints",
            },
            {
                "priority": "P1",
                "task": "Communication radius sweep + Pareto figure",
                "status": "pending",
            },
            {
                "priority": "P1",
                "task": "Ablation bar chart (w/o DG, Temporal, Residual)",
                "status": "pending",
                "data": "results/dsgf/ablation_*",
            },
            {
                "priority": "P2",
                "task": "Complexity analysis (runtime / memory / O(kN))",
                "status": "pending",
            },
            {
                "priority": "P2",
                "task": "Trajectory / communication graph figure",
                "status": "pending",
            },
            {
                "priority": "writing",
                "task": "Fill chapter3_method.tex + chapter4_exp.tex",
                "status": "in_progress",
            },
        ],
    }
    out = ROOT / "results" / "pipeline_status.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    print(f"\n[pipeline] Status -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target = len(SEEDS) * len(METHODS)
    n, missing = count_summaries()
    print(f"[pipeline] Summaries: {n}/{target}")
    if missing:
        print(f"[pipeline] Missing ({len(missing)}): {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}")

    if n < target and not args.force:
        print("[pipeline] Not all runs finished — abort (use --force for partial)")
        sys.exit(1)

    steps_done = []
    report_json = ROOT / "paper" / "tables" / "table1_baseline_5seed.json"

    if run_script("analyze_baseline16_stats.py") != 0:
        sys.exit(1)
    steps_done.append("analyze_baseline16_stats")

    if run_script("plot_baseline16_5seed.py") != 0:
        print("[pipeline] Warning: figure generation failed")
    else:
        steps_done.append("plot_baseline16_5seed")

    export_5seed_csv(report_json, ROOT / "paper" / "tables" / "table1_baseline_5seed.csv")
    steps_done.append("export_table1_5seed_csv")

    # Scalability figure if data exists
    if (ROOT / "results" / "scalability" / "summary.json").exists():
        if run_script("plot_scalability.py") == 0:
            steps_done.append("plot_scalability")

    write_status(report_json, n, steps_done)
    print("\n[pipeline] P0 evidence chain complete.")
    print("  Tables: paper/tables/table1_baseline_5seed.csv")
    print("  Figures: paper/figures/fig2_learning_curve_5seed.png, fig_auc_baseline16.png")
    print("  Next: see results/pipeline_status.json")


if __name__ == "__main__":
    main()
