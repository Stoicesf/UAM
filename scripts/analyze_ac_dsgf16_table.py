"""Assemble Table I++ : baseline16_seeds + AC-DSGF 16UAV × 5 seeds.

CEI = Success / (Comm + ε),  ε = 1e-6
MAPPO has no graph messaging (C≈0) → CEI reported as n/a.

Writes:
  paper/tables/table1_plus_ac.csv
  paper/tables/table1_plus_ac_paper.csv   (human-readable %)
  results/ac_dsgf/uav16/table1_plus_detail.json
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 3407, 2026, 1234, 8888]
BASELINE = ROOT / "results" / "baseline16_seeds"
AC_ROOT = ROOT / "results" / "ac_dsgf" / "uav16"
EPS = 1e-6
METHOD_ORDER = ["mappo", "gat", "transformer", "dsgf", "ac_dsgf"]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v)


def cei(success: float, comm: float) -> float:
    return success / (comm + EPS)


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_from_summary(data: dict) -> dict:
    paper = data.get("paper_metrics", data)
    success = float(paper.get("success", data.get("success", 0.0)))
    collision = float(paper.get("collision", data.get("collision", 0.0)))
    reward = float(paper.get("reward", data.get("reward", 0.0)))
    comm = data.get("communication_cost_mean")
    if comm is None:
        comm = paper.get("communication_cost", 0.0)
    comm = float(comm or 0.0)
    return {
        "success": success,
        "collision": collision,
        "reward": reward,
        "comm_cost": comm,
        "cei": cei(success, comm),
        # legacy alias
        "ce": cei(success, comm),
    }


def collect_method(method: str, root: Path) -> list[dict]:
    rows = []
    for seed in SEEDS:
        if method == "ac_dsgf":
            p = root / f"s{seed}" / "summary.json"
            if not p.exists():
                p = ROOT / "results" / "ac_dsgf" / f"uav16_s{seed}" / "summary.json"
        else:
            p = root / method / f"s{seed}" / "summary.json"
        data = load_summary(p)
        if data is None:
            print(f"[missing] {method} seed={seed}: {p}")
            continue
        m = metrics_from_summary(data)
        m.update({"method": method, "seed": seed, "path": str(p)})
        rows.append(m)
    return rows


def main():
    all_rows = []
    for method in METHOD_ORDER:
        root = AC_ROOT if method == "ac_dsgf" else BASELINE
        all_rows.extend(collect_method(method, root))

    summary = []
    for method in METHOD_ORDER:
        xs = [r for r in all_rows if r["method"] == method]
        if not xs:
            continue
        sm, ss = _mean_std([r["success"] for r in xs])
        cm, cs = _mean_std([r["comm_cost"] for r in xs])
        rem, res = _mean_std([r["reward"] for r in xs])
        col_m, col_s = _mean_std([r["collision"] for r in xs])
        # Recompute CEI from mean S,C for stability; also seed-wise std
        cei_m = cei(sm, cm) if method != "mappo" else float("nan")
        cei_seeds = [r["cei"] for r in xs] if method != "mappo" else []
        _, cei_s = _mean_std(cei_seeds) if cei_seeds else (float("nan"), float("nan"))
        summary.append({
            "method": method,
            "n_seeds": len(xs),
            "success_mean": round(sm, 4),
            "success_std": round(ss, 4),
            "collision_mean": round(col_m, 4),
            "collision_std": round(col_s, 4),
            "reward_mean": round(rem, 4),
            "reward_std": round(res, 4),
            "comm_mean": round(cm, 4),
            "comm_std": round(cs, 4),
            "cei_mean": None if method == "mappo" else round(cei_m, 6),
            "cei_std": None if method == "mappo" else round(cei_s, 6),
        })
        cei_str = "n/a" if method == "mappo" else f"{cei_m:.4f}"
        print(
            f"{method:12} n={len(xs)}  "
            f"S={sm*100:.2f}±{ss*100:.2f}%  "
            f"Col={col_m*100:.1f}%  "
            f"R={rem:.2f}  "
            f"C={cm:.2f}±{cs:.2f}  CEI={cei_str}"
        )

    out_dir = ROOT / "paper" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "table1_plus_ac.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Saved {out_csv}")

    # Paper-facing short table
    paper_rows = []
    for s in summary:
        paper_rows.append({
            "Method": {
                "mappo": "MAPPO",
                "gat": "GAT",
                "transformer": "Transformer",
                "dsgf": "DSGF",
                "ac_dsgf": "AC-DSGF",
            }[s["method"]],
            "Success (%)": f"{100*s['success_mean']:.2f}±{100*s['success_std']:.2f}",
            "Collision (%)": f"{100*s['collision_mean']:.1f}±{100*s['collision_std']:.1f}",
            "Reward": f"{s['reward_mean']:.2f}±{s['reward_std']:.2f}",
            "Comm": f"{s['comm_mean']:.2f}±{s['comm_std']:.2f}",
            "CEI": "—" if s["cei_mean"] is None else f"{s['cei_mean']:.4f}",
        })
    paper_csv = out_dir / "table1_plus_ac_paper.csv"
    with open(paper_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paper_rows[0].keys()))
        w.writeheader()
        w.writerows(paper_rows)
    print(f"Saved {paper_csv}")

    detail = ROOT / "results" / "ac_dsgf" / "uav16" / "table1_plus_detail.json"
    detail.parent.mkdir(parents=True, exist_ok=True)
    detail.write_text(
        json.dumps(
            {"epsilon": EPS, "cei_def": "S/(C+eps)", "per_seed": all_rows, "summary": summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {detail}")


if __name__ == "__main__":
    main()
