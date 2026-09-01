"""Soft communication budget constraint + curriculum schedule (v2e anneal).

Unlike hard Top-K (`budget_layer.py`), this is a training loss that
pulls average gate mass toward a target without silence-collapse
from oversized λ_comm:

    L_b = (C_t / C_target - 1)^2

Annealing (curriculum):
  Stage 1  t < start_step:     λ_b=0  (free exploration)
  Stage 2  start→end:          linear target & λ_b
  Stage 3  t ≥ end_step:       fixed end_target / end_λ_b
"""

from __future__ import annotations

from typing import Any

import torch

from models.communication.cost import communication_cost


def soft_budget_loss(
    g: torch.Tensor,
    target_comm: float,
    lambda_b: float = 1.0,
) -> torch.Tensor:
    """Quadratic budget deviation on mean directed gate mass C = Σ g_ij.

    Args:
        g: (B, N, N) or (N, N) soft gates
        target_comm: desired mean C (same units as communication_cost)
        lambda_b: loss weight

    Returns:
        scalar λ_b · (C / C_target − 1)²
    """
    if lambda_b <= 0:
        return g.new_zeros(())
    c = communication_cost(g, reduction="mean")
    target = max(float(target_comm), 1e-6)
    return lambda_b * ((c / target) - 1.0) ** 2


def _budget_block(guidance_cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept nested `budget:` or flat `budget_*` keys."""
    block = guidance_cfg.get("budget")
    if isinstance(block, dict) and block:
        return block
    return {
        "enabled": bool(guidance_cfg.get("budget_anneal", False)),
        "schedule": guidance_cfg.get("budget_schedule", "linear"),
        "start_step": guidance_cfg.get("budget_start_step", 20_000),
        "end_step": guidance_cfg.get("budget_end_step", 70_000),
        "start_target": guidance_cfg.get("budget_start_target", 2.0),
        "end_target": guidance_cfg.get("budget_end_target", 1.0),
        "lambda_start": guidance_cfg.get("budget_lambda_start", 0.0),
        "lambda_end": guidance_cfg.get(
            "budget_lambda_end", guidance_cfg.get("lambda_budget", 0.02)
        ),
    }


def scheduled_budget(
    step: int,
    guidance_cfg: dict[str, Any],
    *,
    fallback_lambda: float = 0.0,
    fallback_target: float = 0.5,
    utility_warmup_steps: int = 0,
) -> tuple[float, float]:
    """Return (lambda_b, target_comm) at training step.

    If budget annealing is disabled, falls back to fixed λ_b / target
    (after optional utility warm-up with λ_b=0).
    """
    block = _budget_block(guidance_cfg)
    enabled = bool(block.get("enabled", False))

    if not enabled:
        if utility_warmup_steps > 0 and step < utility_warmup_steps:
            return 0.0, float(fallback_target)
        return float(fallback_lambda), float(fallback_target)

    start_step = int(block.get("start_step", 20_000))
    end_step = int(block.get("end_step", 70_000))
    start_target = float(block.get("start_target", 2.0))
    end_target = float(block.get("end_target", 1.0))
    lambda_start = float(block.get("lambda_start", 0.0))
    lambda_end = float(block.get("lambda_end", 0.02))

    # Stage 1 — free communication (also respect utility warm-up)
    if step < start_step:
        return 0.0, start_target

    # Stage 3 — fixed final budget
    if step >= end_step or end_step <= start_step:
        return lambda_end, end_target

    # Stage 2 — linear compression
    t = (step - start_step) / float(end_step - start_step)
    t = max(0.0, min(1.0, t))
    target = start_target + t * (end_target - start_target)
    lambda_b = lambda_start + t * (lambda_end - lambda_start)
    return float(lambda_b), float(target)
