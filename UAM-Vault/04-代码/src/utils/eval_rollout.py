"""Post-training rollout evaluation for paper metrics."""

from __future__ import annotations

import torch
from torchrl.envs.utils import step_mdp

from reward.guidance_reward import guidance_alignment_reward


def _goal_distance_from_obs(obs: torch.Tensor) -> torch.Tensor:
    """Navigation obs: [pos(2), vel(2), goal_rel(2), lidar...]."""
    if obs.shape[-1] >= 6:
        goal_rel = obs[..., 4:6]
    else:
        goal_rel = obs[..., -2:]
    return goal_rel.norm(dim=-1)


def evaluate_policy(
    env,
    policy,
    num_episodes: int = 32,
    success_threshold: float = 0.3,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Run evaluation rollouts and return paper metrics.

    Important: must call ``step_mdp`` after ``env.step`` (TorchRL). Diagnostic
    scripts already did this; older Table-I path did not — do not regress.
    """
    policy.eval()
    successes = []
    collisions = []
    path_lengths = []
    episode_lengths = []
    alignments = []
    rewards = []
    comm_masses = []

    steps = max_steps or getattr(env, "max_steps", 128) or 128

    with torch.no_grad():
        for _ in range(num_episodes):
            td = env.reset()
            ep_reward = 0.0
            ep_path = 0.0
            ep_collision = 0.0
            ep_align = []
            ep_success = 0.0
            ep_steps = 0
            ep_comm = []

            for _ in range(steps):
                td = policy(td)
                td = env.step(td)
                ep_steps += 1

                obs = td.get(("next", "agents", "observation"))
                dist = _goal_distance_from_obs(obs)
                ep_success = (dist < success_threshold).float().mean().item()

                actions = td.get(("agents", "action"))
                ep_path += actions[..., :2].norm(dim=-1).mean().item()

                info = td.get(("next", "agents", "info"), default=None)
                if info is not None and "agent_collisions" in info.keys():
                    coll = info.get("agent_collisions")
                    ep_collision = max(ep_collision, (coll < 0).float().mean().item())

                phi = td.get(("agents", "phi"))
                if phi is not None and actions is not None:
                    ep_align.append(guidance_alignment_reward(actions, phi).mean().item())

                rew = td.get(("next", "agents", "reward"))
                ep_reward += rew.mean().item()

                try:
                    from algorithms.guided.mappo_guided import ACGuideAdapter

                    for m in policy.modules():
                        if isinstance(m, ACGuideAdapter):
                            ep_comm.append(float(m.last_soft_edges))
                            break
                except Exception:
                    pass

                done = td.get(("next", "done"))
                if done is not None and bool(done.any()):
                    break
                td = step_mdp(td)

            successes.append(ep_success)
            collisions.append(ep_collision)
            path_lengths.append(ep_path)
            episode_lengths.append(ep_steps)
            rewards.append(ep_reward)
            if ep_align:
                alignments.append(sum(ep_align) / len(ep_align))
            if ep_comm:
                comm_masses.append(sum(ep_comm) / len(ep_comm))

    policy.train()
    out = {
        "reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "success": sum(successes) / len(successes) if successes else 0.0,
        "collision": sum(collisions) / len(collisions) if collisions else 0.0,
        "path_length": sum(path_lengths) / len(path_lengths) if path_lengths else 0.0,
        "episode_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0,
        "alignment": sum(alignments) / len(alignments) if alignments else 0.0,
    }
    if comm_masses:
        out["communication_cost"] = sum(comm_masses) / len(comm_masses)
    return out
