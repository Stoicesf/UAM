"""Message discrepancy: ε_G = ||M(G*) - M(G_t)||."""

from __future__ import annotations

import torch


def message_l2(m_full: torch.Tensor, m_sparse: torch.Tensor) -> torch.Tensor:
    """Per-batch Frobenius / flattened L2 over agent×feature dims → (B,)."""
    diff = m_full - m_sparse
    return diff.reshape(diff.shape[0], -1).norm(dim=-1)


def message_mean_l2(m_full: torch.Tensor, m_sparse: torch.Tensor) -> float:
    return float(message_l2(m_full, m_sparse).mean().item())
