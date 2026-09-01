"""P1-3 — Aggregate communication sweep results + Pareto figure.

Usage:
  python scripts/eval_communication.py --dry-run
  python scripts/eval_communication.py --plot
  python scripts/eval_communication.py --profile full --plot

Reads results/communication/{run_name}/summary.json
Outputs:
  results/communication/summary.csv
  paper/tables/table4_communication.csv
  paper/figures/fig6_communication_pareto.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "communication" / "manifest.json"
RESULTS_DIR = ROOT / "results" / "communication"


def load_manifest() -> dict:
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def comm_efficiency(success: float, comm_cost: float) -> float:
    if comm_cost <= 0:
        return success * 100.0 if success > 0 else 0.0
    return success / comm_cost


def collect_rows(profile: str, dry_run: bool) -> list[dict]:
    manifest = load_manifest()
    runs = manifest["profiles"].get(profile, [])
    rows = []

    for spec in runs:
        run_name = spec["run_name"]
        method = spec["method"]
        radius_key = spec["radius"]
        summary_path = RESULTS_DIR / run_name / "summary.json"

        if dry_run:
            exists = summary_path.exists()
            print(f"[dry-run] {run_name}: exists={exists}")
            continue

        if not summary_path.exists():
            print(f"[skip] {run_name}: no summary.json")
            continue

        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        cfg_path = RESULTS_DIR / run_name / "config_resolved.yaml"
        comm_sweep = summary.get("comm_sweep") or {}
        if not comm_sweep and cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                comm_sweep = yaml.safe_load(f).get("comm_sweep", {})

        paper = summary.get("paper_metrics", summary)
        success = float(paper.get("success", summary.get("success", 0.0)))

        if method == "mappo":
            comm_cost = float(manifest["methods"]["mappo"].get("implicit_comm_cost", 0.0))
        else:
            comm_cost = float(
                summary.get("communication_cost_mean")
                or paper.get("communication_cost")
                or summary.get("communication_cost", 0.0)
            )

        radius_label = radius_key
        if comm_sweep.get("radius_label"):
            radius_label = comm_sweep["radius_label"]

        row = {
            "method": method,
            "radius": radius_label,
            "radius_key": radius_key,
            "comm_radius": comm_sweep.get("comm_radius", summary.get("comm_radius")),
            "success": round(success, 4),
            "success_pct": round(success * 100, 2),
            "comm_cost": round(comm_cost, 4),
            "efficiency": round(comm_efficiency(success, comm_cost), 6),
            "collision": round(float(paper.get("collision", 0.0)), 4),
            "run_name": run_name,
            "checkpoint": str((RESULTS_DIR / run_name / "checkpoints" / "final.pt").as_posix()),
        }
        rows.append(row)
        print(
            f"{method} R={radius_label}: success={row['success_pct']:.2f}% "
            f"comm={comm_cost:.2f} eta={row['efficiency']:.4f}"
        )

    return rows


def export_tables(rows: list[dict]):
    if not rows:
        print("[warn] no rows to export")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "radius", "radius_key", "comm_radius",
        "success", "success_pct", "comm_cost", "efficiency",
        "collision", "run_name", "checkpoint",
    ]

    summary_csv = RESULTS_DIR / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {summary_csv}")

    paper_dir = ROOT / "paper" / "tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    table_fields = ["method", "radius", "success_pct", "comm_cost", "efficiency", "collision"]
    table_csv = paper_dir / "table4_communication.csv"
    with open(table_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=table_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in table_fields})
    print(f"Saved {table_csv}")

    json_path = paper_dir / "table4_communication.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {json_path}")


def plot_pareto(rows: list[dict]):
    if not rows:
        return

    colors = {"mappo": "#4C72B0", "gat": "#DD8452", "dsgf": "#C44E52"}
    markers = {"mappo": "s", "gat": "^", "dsgf": "o"}

    plt.figure(figsize=(8, 5.5))
    for method in sorted({r["method"] for r in rows}):
        pts = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda x: x["comm_cost"],
        )
        xs = [p["comm_cost"] for p in pts]
        ys = [p["success_pct"] for p in pts]
        plt.plot(
            xs, ys,
            marker=markers.get(method, "o"),
            linewidth=2,
            markersize=8,
            label=method.upper(),
            color=colors.get(method),
        )
        for p in pts:
            plt.annotate(
                f"R={p['radius']}",
                (p["comm_cost"], p["success_pct"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
                color=colors.get(method),
            )

    plt.xlabel("Communication Cost (edges / timestep)")
    plt.ylabel("Success Rate (%)")
    plt.title("Communication Efficiency: Success vs. Communication Cost")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = ROOT / "paper" / "figures" / "fig6_communication_pareto.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="fast", choices=["full", "fast", "smoke"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = collect_rows(args.profile, dry_run=args.dry_run)
    if args.dry_run:
        return

    export_tables(rows)
    if args.plot and rows:
        plot_pareto(rows)


if __name__ == "__main__":
    main()
