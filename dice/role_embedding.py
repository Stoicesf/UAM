"""Compact role embeddings for low-bandwidth messaging."""

from __future__ import annotations

import torch
import torch.nn as nn


class RoleEmbedding(nn.Module):
    def __init__(self, n_roles: int = 3, dim: int = 8):
        super().__init__()
        self.emb = nn.Embedding(n_roles, dim)

    def forward(self, roles: torch.Tensor) -> torch.Tensor:
        return self.emb(roles.clamp(0, self.emb.num_embeddings - 1))


def self_check() -> None:
    m = RoleEmbedding(3, 8)
    x = m(torch.tensor([0, 1, 2]))
    assert x.shape == (3, 8)
    print("role_embedding: OK")


if __name__ == "__main__":
    self_check()
