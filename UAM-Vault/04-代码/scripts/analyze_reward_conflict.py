"""Diagnose reward conflict from metrics.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def analyze(metrics_path: Path):
    with open(metrics_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No metrics found.")
        return

    first, last = rows[0], rows[-1]
    print(f"=== Reward Conflict Analysis: {metrics_path.parent.name} ===\n")
    print(f"{'Metric':<20} {'Start':>10} {'End':>10} {'Trend':>10}")
    print("-" * 52)

    for key in ("reward_guide", "reward_goal", "reward_collision", "action_alignment", "success_rate"):
        s = float(first.get(key, 0))
        e = float(last.get(key, 0))
        trend = "UP" if e > s + 1e-6 else ("DOWN" if e < s - 1e-6 else "FLAT")
        print(f"{key:<20} {s:>10.4f} {e:>10.4f} {trend:>10}")

    guide_up = float(last["reward_guide"]) > float(first["reward_guide"])
    goal_down = float(last["reward_goal"]) < float(first["reward_goal"])
    print()
    if guide_up and goal_down:
        print("DIAGNOSIS: Guide reward UP + Goal reward DOWN")
        print("  -> Guidance is competing with navigation objective.")
        print("  -> Reduce guidance_reward_coef or add warmup.")
    elif guide_up and not goal_down:
        print("DIAGNOSIS: Guide and Goal both improving — healthy.")
    else:
        print("DIAGNOSIS: Mixed signals — check TensorBoard reward/ curves.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/guide/guide_gate3_v1/metrics.csv")
    args = parser.parse_args()
    analyze(Path(args.run))


if __name__ == "__main__":
    main()
