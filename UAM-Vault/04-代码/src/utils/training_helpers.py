"""Training helpers — milestone checkpoints and metric aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


MILESTONE_FRAMES = [20_000, 40_000, 60_000, 80_000, 100_000]


def maybe_save_milestones(
    global_step: int,
    saved: set[int],
    components: Any,
    ckpt_dir: Path,
    reward: float,
    save_fn,
) -> set[int]:
    for milestone in MILESTONE_FRAMES:
        if global_step >= milestone and milestone not in saved:
            path = ckpt_dir / f"checkpoint_{milestone // 1000}k.pt"
            save_fn(components, str(path), global_step, reward)
            saved.add(milestone)
    return saved


def aggregate_training_metrics(metrics_history: list[dict]) -> dict[str, float]:
    """Aggregate per-iteration metrics for summary.json."""
    if not metrics_history:
        return {}

    def mean_key(key: str) -> float:
        vals = [m[key] for m in metrics_history if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    def last_key(key: str) -> float:
        for m in reversed(metrics_history):
            if key in m:
                return m[key]
        return 0.0

    tail = metrics_history[-max(1, len(metrics_history) // 5) :]

    def tail_mean(key: str) -> float:
        vals = [m[key] for m in tail if key in m]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "reward_mean": mean_key("episode_reward_mean"),
        "reward_final": last_key("episode_reward_mean"),
        "success_mean": mean_key("success_rate"),
        "success_final": tail_mean("success_rate"),
        "collision_mean": mean_key("collision_rate"),
        "collision_final": tail_mean("collision_rate"),
        "alignment_mean": mean_key("action_alignment"),
        "alignment_final": tail_mean("action_alignment"),
        "path_length_mean": mean_key("path_length"),
        "path_length_final": tail_mean("path_length"),
        "communication_mean": mean_key("sparse_edges"),
        "communication_final": tail_mean("sparse_edges"),
    }


def build_paper_summary(
    reward_curve: list[float],
    metrics_history: list[dict],
    eval_stats: dict | None = None,
) -> dict:
    agg = aggregate_training_metrics(metrics_history)
    eval_stats = eval_stats or {}

    return {
        "reward": round(eval_stats.get("reward", agg.get("reward_final", 0.0)), 4),
        "success": round(eval_stats.get("success", agg.get("success_final", 0.0)), 4),
        "collision": round(eval_stats.get("collision", agg.get("collision_final", 0.0)), 4),
        "path_length": round(eval_stats.get("path_length", agg.get("path_length_final", 0.0)), 4),
        "episode_length": round(eval_stats.get("episode_length", 0.0), 4),
        "alignment": round(eval_stats.get("alignment", agg.get("alignment_final", 0.0)), 4),
        # Prefer eval-time Comm when available; never silently mix training averages
        "communication_cost": round(
            eval_stats.get("communication_cost", agg.get("communication_mean", 0.0)), 4
        ),
        "communication_cost_source": (
            "eval" if "communication_cost" in eval_stats else "train_log_mean"
        ),
        "max_reward": round(max(reward_curve), 4) if reward_curve else 0.0,
        "num_iterations": len(reward_curve),
    }
