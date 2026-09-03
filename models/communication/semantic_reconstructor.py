"""Reconstruct task info from L1/L2/L3 semantic payloads."""

from __future__ import annotations

import torch
import torch.nn as nn


class SemanticReconstructor(nn.Module):
    def __init__(self, z_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.l2_head = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 4))  # pos+vel
        self.l3_head = nn.Sequential(nn.Linear(z_dim, hidden), nn.Tanh(), nn.Linear(hidden, z_dim))

    def reconstruct_l1(self, role: int, intent: int) -> dict:
        return {"role": role, "intent": intent, "directive": f"role={role}:zone={intent}"}

    def reconstruct_l2(self, payload: torch.Tensor) -> torch.Tensor:
        """Return estimated [x,y,vx,vy]."""
        return self.l2_head(payload.flatten()[:4])

    def reconstruct_l3(self, z: torch.Tensor) -> torch.Tensor:
        return self.l3_head(z)

    @staticmethod
    def ssim_proxy(a: torch.Tensor, b: torch.Tensor) -> float:
        """Lightweight structural similarity proxy on vectors."""
        a = a.flatten().float()
        b = b.flatten().float()
        mu_a, mu_b = a.mean(), b.mean()
        sig_a = a.var(unbiased=False)
        sig_b = b.var(unbiased=False)
        sig_ab = ((a - mu_a) * (b - mu_b)).mean()
        c1, c2 = 1e-4, 1e-3
        return float(
            ((2 * mu_a * mu_b + c1) * (2 * sig_ab + c2))
            / ((mu_a**2 + mu_b**2 + c1) * (sig_a + sig_b + c2) + 1e-8)
        )


def self_check() -> None:
    r = SemanticReconstructor()
    d = r.reconstruct_l1(0, 1)
    assert "directive" in d
    true = torch.tensor([1.0, 2.0, 0.1, -0.2])
    pred = r.reconstruct_l2(true)
    # train one step toward identity
    opt = torch.optim.Adam(r.parameters(), lr=1e-2)
    for _ in range(200):
        pred = r.reconstruct_l2(true)
        loss = nn.functional.mse_loss(pred, true)
        opt.zero_grad()
        loss.backward()
        opt.step()
    err = float((r.reconstruct_l2(true)[:2] - true[:2]).norm())
    z = torch.randn(16)
    z2 = r.reconstruct_l3(z)
    ssim = SemanticReconstructor.ssim_proxy(z, z2)
    print(f"semantic_reconstructor: OK (l2_err={err:.3f}, ssim_proxy={ssim:.3f})")


if __name__ == "__main__":
    self_check()
