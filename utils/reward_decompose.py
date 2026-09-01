"""Decomposed reward metrics from VMAS tensordict + guide shaping."""

from __future__ import annotations

import torch

from reward.guidance_reward import guidance_alignment_reward


def extract_vmas_reward_parts(tensordict_data, env) -> dict[str, float]:
    """Extract goal / collision components from VMAS info."""
    info = tensordict_data.get(("next", "agents", "info"), default=None)
    if info is None:
        total = tensordict_data.get(("next", env.reward_key))
        return {
            "reward_total_env": total.mean().item(),
            "reward_goal": 0.0,
            "reward_collision": 0.0,
            "reward_final": 0.0,
        }

    pos_rew = info.get("pos_rew")
    final_rew = info.get("final_rew")
    collision = info.get("agent_collisions")

    goal = pos_rew.mean().item() if pos_rew is not None else 0.0
    final = final_rew.mean().item() if final_rew is not None else 0.0
    coll = collision.mean().item() if collision is not None else 0.0
    total_env = goal + final + coll

    success_rate = 0.0
    if final_rew is not None:
        success_rate = (final_rew > 0).float().mean().item()

    collision_rate = 0.0
    if collision is not None:
        collision_rate = (collision < 0).float().mean().item()

    return {
        "reward_goal": goal + final,
        "reward_collision": coll,
        "reward_final": final,
        "reward_total_env": total_env,
        "success_rate": success_rate,
        "collision_rate": collision_rate,
    }


def compute_guide_reward_metrics(
    tensordict_data,
    guide_coef: float,
) -> dict[str, float]:
    actions = tensordict_data.get(("agents", "action"), default=None)
    phi = tensordict_data.get(("agents", "phi"), default=None)
    if actions is None or phi is None:
        return {
            "reward_guide": 0.0,
            "action_alignment": 0.0,
        }
    alignment = guidance_alignment_reward(actions, phi)
    guide = guide_coef * alignment
    return {
        "reward_guide": guide.mean().item(),
        "action_alignment": alignment.mean().item(),
    }


def compute_communication_cost(num_agents: int) -> float:
    """Stage 1 placeholder: full graph cost (no gating yet)."""
    if num_agents <= 1:
        return 0.0
    return num_agents * (num_agents - 1) / 2.0
