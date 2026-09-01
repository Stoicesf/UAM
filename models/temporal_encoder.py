"""Temporal memory — GRU over spatial node embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """z_i^t = GRU(h_i^t, z_i^{t-1}).

    Gate-1 / single-step forward uses zero initial hidden state.
    Runner can pass z_prev for multi-step memory (Week 4+).
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        z_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h: (B, N, D) spatial embeddings
            z_prev: optional (B, N, H) previous hidden state
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)

        b, n, d = h.shape
        if z_prev is None:
            z_prev = h.new_zeros(b, n, self.hidden_dim)

        h_flat = h.reshape(b * n, d)
        z_flat = z_prev.reshape(b * n, self.hidden_dim)
        z_new = self.gru(h_flat, z_flat)
        return z_new.reshape(b, n, self.hidden_dim)
