"""Communication cost metrics for sparse dynamic graphs."""

from __future__ import annotations

import torch

from guidance.graph_builder import build_adjacency


def compute_sparse_communication_stats(
    observations: torch.Tensor,
    comm_radius: float,
) -> dict[str, float]:
    """Estimate per-step communication from agent positions in obs.

    C_t = sum_{i,j} A_ij  (undirected edges, no self-loops)

    Args:
        observations: (B, N, obs_dim) or (N, obs_dim)
    """
    if observations.dim() == 2:
        observations = observations.unsqueeze(0)
    elif observations.dim() == 4:
        # (num_envs, time, n_agents, dim) from batched rollout
        observations = observations.reshape(
            -1, observations.shape[-2], observations.shape[-1]
        )
    elif observations.dim() != 3:
        observations = observations.reshape(-1, observations.shape[-2], observations.shape[-1])

    positions = observations[..., :2]
    adj = build_adjacency(positions, comm_radius)
    edges_per_env = adj.sum(dim=(-2, -1))
    n_agents = observations.shape[-2]
    full_graph_edges = float(n_agents * (n_agents - 1))

    sparse_mean = edges_per_env.mean().item()
    return {
        "sparse_edges": sparse_mean,
        "full_graph_edges": full_graph_edges,
        "communication_ratio": sparse_mean / max(full_graph_edges, 1.0),
        "communication_cost": sparse_mean,
    }
