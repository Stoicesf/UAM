"""Guidance reward coefficient scheduling — warmup, ramp, adaptive decay, alignment."""

from __future__ import annotations

import math


def compute_guidance_coef(
    global_step: int,
    total_frames: int,
    max_coef: float,
    warmup_fraction: float = 0.2,
    use_warmup: bool = False,
    schedule: str = "fixed",
    min_coef: float = 0.02,
    decay_k: float = 3.0,
    alignment: float | None = None,
) -> float:
    """Compute λ_t for guidance reward shaping.

    Schedules:
        fixed: constant max_coef
        warmup_ramp: λ=0 during warmup, linear ramp to max_coef
        adaptive_decay: warmup → λ_max, then λ_max·exp(-k·progress), floor min_coef
        adaptive_align: adaptive_decay × (1 - alignment) to curb over-guidance
    """
    if max_coef <= 0:
        return 0.0

    if schedule == "fixed" and not use_warmup:
        return max_coef

    warmup_frames = int(total_frames * warmup_fraction) if use_warmup else 0
    if use_warmup and global_step <= warmup_frames:
        return 0.0

    post_total = max(total_frames - warmup_frames, 1)
    progress = min(1.0, max(0.0, (global_step - warmup_frames) / post_total))

    if schedule in ("warmup_ramp", "fixed"):
        if use_warmup:
            return max_coef * progress
        return max_coef

    if schedule in ("adaptive_decay", "adaptive_align"):
        coef = max_coef * math.exp(-decay_k * progress)
        coef = max(min_coef, coef)
    else:
        coef = max_coef * progress if use_warmup else max_coef

    if schedule == "adaptive_align" and alignment is not None:
        coef *= max(0.0, 1.0 - min(max(alignment, 0.0), 1.0))

    return coef


def compute_residual_beta(
    global_step: int,
    total_frames: int,
    beta_max: float = 1.0,
    beta_min: float = 0.0,
    warmup_fraction: float = 0.2,
    use_warmup: bool = True,
    decay_k: float = 3.0,
) -> float:
    """β_t for residual guidance: high early (explore with Φ), → 0 late (autonomous NavRL)."""
    if beta_max <= 0:
        return beta_min

    warmup_frames = int(total_frames * warmup_fraction) if use_warmup else 0
    if use_warmup and global_step <= warmup_frames:
        return beta_max * (global_step / max(warmup_frames, 1))

    post_total = max(total_frames - warmup_frames, 1)
    progress = min(1.0, max(0.0, (global_step - warmup_frames) / post_total))
    beta = beta_max * math.exp(-decay_k * progress)
    return max(beta_min, beta)
