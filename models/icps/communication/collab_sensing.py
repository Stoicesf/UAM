"""Collaborative sensing request/response + simple consensus."""

from __future__ import annotations

import torch


def request_collab(
    need_help: torch.Tensor,
    radius_mask: torch.Tensor,
) -> torch.Tensor:
    """(B,N,N) request matrix: i→j if i needs help and j in range."""
    if need_help.dim() == 1:
        need_help = need_help.unsqueeze(0)
    return need_help.unsqueeze(-1) * radius_mask


def respond_collab(
    request: torch.Tensor,
    local_flags: torch.Tensor,
) -> torch.Tensor:
    """Neighbors share obstacle/event flags. local_flags: (B,N) or (B,N,F)."""
    if local_flags.dim() == 2:
        local_flags = local_flags.unsqueeze(-1)
    # response payload from j to i: request_ij * flag_j
    # out[i] = max over j of request_ij * flag_j
    resp = request.unsqueeze(-1) * local_flags.unsqueeze(1)  # B,N,N,F
    fused, _ = resp.max(dim=2)
    return fused.squeeze(-1) if fused.shape[-1] == 1 else fused


def consensus_vote(
    node_flags: torch.Tensor,
    radius_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Majority vote among neighbors; return agreed flags + agreement rate."""
    if node_flags.dim() == 1:
        node_flags = node_flags.unsqueeze(0)
    # average neighbor flags
    deg = radius_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    neigh_mean = (radius_mask @ node_flags.unsqueeze(-1)).squeeze(-1) / deg.squeeze(-1)
    agreed = (neigh_mean >= threshold).float()
    # consistency: fraction of nodes matching global majority
    global_maj = (node_flags.mean(dim=-1, keepdim=True) >= threshold).float()
    consistency = float((agreed == global_maj).float().mean())
    return agreed, consistency


def self_check() -> None:
    n = 4
    mask = torch.ones(1, n, n) - torch.eye(n)
    need = torch.tensor([[1.0, 0, 0, 1.0]])
    req = request_collab(need, mask)
    flags = torch.tensor([[1.0, 0, 1.0, 0]])
    fused = respond_collab(req, flags)
    assert fused.shape == (1, n)
    _, c = consensus_vote(flags, mask)
    assert 0.0 <= c <= 1.0
    print(f"collab_sensing: OK (consensus={c:.2f})")


if __name__ == "__main__":
    self_check()
