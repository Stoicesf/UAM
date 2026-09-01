"""Load trained MAPPO / guided policies for zero-shot evaluation."""

from __future__ import annotations

from pathlib import Path

import torch

from algorithms.baseline.mappo import build_mappo
from algorithms.baseline.runner import load_checkpoint as load_mappo_ckpt
from algorithms.guided.mappo_guided import build_guided_mappo


def resolve_checkpoint(primary: str | None, fallback: str | None = None) -> Path:
    p = Path(primary) if primary else None
    if p and p.exists():
        return p
    if fallback:
        fb = Path(fallback)
        if fb.exists():
            return fb
    raise FileNotFoundError(f"Checkpoint not found: {primary} (fallback={fallback})")


def load_policy_for_eval(exp_cfg: dict, checkpoint: str | Path):
    """Build env+policy, load weights, return (env, policy) in eval mode."""
    train_cfg = exp_cfg["train"]
    env_cfg = exp_cfg["env"]
    guidance_cfg = exp_cfg.get("guidance", {"mode": "none"})
    mode = guidance_cfg.get("mode", "none")

    if mode == "none":
        components = build_mappo(train_cfg, env_cfg)
        load_mappo_ckpt(components, str(checkpoint))
        policy = components.policy
    else:
        components = build_guided_mappo(train_cfg, env_cfg, guidance_cfg)
        load_mappo_ckpt(components, str(checkpoint))
        policy = components.policy

    policy.eval()
    return components.env, policy
