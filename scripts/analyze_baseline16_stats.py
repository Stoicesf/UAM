"""Aggregate 5-seed baseline16 stats: mean±std, 95% CI, AULC, significance tests.

Usage:
  python scripts/analyze_baseline16_stats.py
  python scripts/analyze_baseline16_stats.py --out paper/tables/table1_baseline_5seed.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 3407, 2026, 1234, 8888]
METHODS = ["mappo", "gat", "transformer", "dsgf"]
FRAMES_PER_BATCH = 2048


def load_run(method: str, seed: int) -> dict | None:
    p = ROOT / "results" / "baseline16_seeds" / method / f"s{seed}" / "summary.json"
    if not p.exists():
        legacy = ROOT / "results" / "baseline16" / method / "summary.json"
        if seed == 42 and legacy.exists():
            with open(legacy, encoding="utf-8") as f:
                return json.load(f)
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def compute_auc(curve: list[float], step: int = FRAMES_PER_BATCH) -> float:
    if not curve:
        return 0.0
    x = np.arange(1, len(curve) + 1) * step
    y = np.array(curve, dtype=np.float64)
    return float(np.trapezoid(y, x) / (x[-1] - x[0] + step))


def ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        v = values[0] if values else 0.0
        return v, v
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    half = 1.96 * std / math.sqrt(n)
    return mean - half, mean + half


def try_ttest(a: list[float], b: list[float]) -> dict:
    try:
        from scipy import stats

        if len(a) < 2 or len(b) < 2:
            return {"t_stat": None, "p_value": None}
        t, p = stats.ttest_ind(a, b, equal_var=False)
        return {"t_stat": float(t), "p_value": float(p)}
    except ImportError:
        return {"t_stat": None, "p_value": None, "note": "scipy not installed"}


def try_wilcoxon(a: list[float], b: list[float]) -> dict:
    try:
        from scipy import stats

        if len(a) != len(b) or len(a) < 2:
            return {"statistic": None, "p_value": None}
        w, p = stats.wilcoxon(a, b)
        return {"statistic": float(w), "p_value": float(p)}
    except ImportError:
        return {"statistic": None, "p_value": None, "note": "scipy not installed"}


def aggregate():
    rows = []
    per_method: dict[str, dict[str, list[float]]] = {
        m: {"success": [], "collision": [], "reward": [], "aulc": []} for m in METHODS
    }

    for method in METHODS:
        seed_runs = []
        for seed in SEEDS:
            s = load_run(method, seed)
            if s is None:
                continue
            pm = s.get("paper_metrics", s)
            seed_runs.append({"seed": seed, **pm})
            per_method[method]["success"].append(pm.get("success", 0.0))
            per_method[method]["collision"].append(pm.get("collision", 0.0))
            per_method[method]["reward"].append(pm.get("reward", 0.0))
            curve = s.get("reward_curve", [])
            per_method[method]["aulc"].append(compute_auc(curve))

        vals = per_method[method]
        n = len(vals["success"])
        if n == 0:
            rows.append({"method": method, "n_seeds": 0})
            continue

        def ms(key: str) -> dict:
            arr = vals[key]
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            lo, hi = ci95(arr)
            return {"mean": round(m, 4), "std": round(s, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4)}

        rows.append({
            "method": method,
            "n_seeds": n,
            "seeds": [r["seed"] for r in seed_runs],
            "success": ms("success"),
            "collision": ms("collision"),
            "reward": ms("reward"),
            "aulc": ms("aulc"),
            "per_seed": seed_runs,
        })

    dsgf_s = per_method["dsgf"]["success"]
    mappo_s = per_method["mappo"]["success"]
    tests = {
        "dsgf_vs_mappo_ttest": try_ttest(dsgf_s, mappo_s),
        "dsgf_vs_mappo_wilcoxon_paired": try_wilcoxon(dsgf_s, mappo_s),
    }
    return {"methods": rows, "significance": tests}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/tables/table1_baseline_5seed.json")
    args = parser.parse_args()

    report = aggregate()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== 5-Seed Baseline16 (mean ± std) ===")
    print(f"{'Method':<14} {'Success':>16} {'AULC':>16} {'n':>3}")
    print("-" * 52)
    for row in report["methods"]:
        if row.get("n_seeds", 0) == 0:
            print(f"{row['method']:<14} {'—':>16} {'—':>16} {0:>3}")
            continue
        s = row["success"]
        a = row["aulc"]
        print(
            f"{row['method']:<14} "
            f"{s['mean']:.4f}±{s['std']:.4f} "
            f"{a['mean']:.2f}±{a['std']:.2f} "
            f"{row['n_seeds']:>3}"
        )

    sig = report["significance"]["dsgf_vs_mappo_ttest"]
    if sig.get("p_value") is not None:
        print(f"\nDSGF vs MAPPO t-test: p={sig['p_value']:.4f}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
