"""Lightweight VAE semantic encoder (obs → z_dim≤16)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticEncoder(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int = 16, hidden: int = 64):
        super().__init__()
        assert z_dim <= 16
        self.z_dim = z_dim
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, obs_dim),
        )

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.enc(obs)
        mu, logvar = self.mu(h), self.logvar(h)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(obs)
        recon = self.dec(z)
        return z, recon, mu, logvar

    def loss(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        z, recon, mu, logvar = self.forward(obs)
        recon_loss = nn.functional.mse_loss(recon, obs)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon_loss + 0.01 * kl
        # relative recon error vs obs energy
        rel = float((recon - obs).norm() / (obs.norm() + 1e-8))
        return total, {"recon": float(recon_loss), "kl": float(kl), "rel_err": rel}


def pack_l1(role: int, intent: int, n_roles: int = 5) -> torch.Tensor:
    """L1 semantic primitive: [role one-hot, intent scalar]."""
    oh = torch.nn.functional.one_hot(torch.tensor(role), n_roles).float()
    return torch.cat([oh, torch.tensor([float(intent)])])


def pack_l2(target_pos: torch.Tensor, target_vel: torch.Tensor) -> torch.Tensor:
    return torch.cat([target_pos.flatten()[:2], target_vel.flatten()[:2]])


def self_check() -> None:
    m = SemanticEncoder(32, 16)
    obs = torch.randn(8, 32)
    loss, stats = m.loss(obs)
    assert m.z_dim <= 16
    loss.backward()
    print(f"semantic_encoder: OK (rel_err={stats['rel_err']:.4f})")


if __name__ == "__main__":
    self_check()
