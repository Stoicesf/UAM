"""Reshape flattened multi-agent observations to (B, N, D)."""

from __future__ import annotations

import torch


def reshape_multiagent_obs(obs: torch.Tensor, n_agents: int) -> torch.Tensor:
    """(B*N, D) | (B, N, D) | (B, T, N, D) → (B, N, D)."""
    if obs.dim() == 4:
        # Collector stores (num_envs, time, n_agents, dim) — use latest step
        obs = obs[:, -1]
    if obs.dim() == 3:
        return obs
    if obs.dim() == 2:
        if n_agents > 0 and obs.shape[0] % n_agents == 0:
            return obs.view(-1, n_agents, obs.shape[-1])
        return obs.unsqueeze(0)
    raise ValueError(f"Unexpected obs shape {tuple(obs.shape)}")
