"""RISE disturbance-rejection compensator (2D)."""

from __future__ import annotations

import torch


class RISEController:
    def __init__(
        self,
        alpha: float = 10.0,
        beta: float = 0.3,
        kv: float = 2.5,
        kp: float = 2.0,
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kv = float(kv)
        self.kp = float(kp)
        self.integral: torch.Tensor | None = None

    def reset(self) -> None:
        self.integral = None

    def compute(
        self,
        e_x: torch.Tensor,
        e_x_dot: torch.Tensor,
        dt: float = 0.05,
    ) -> torch.Tensor:
        """Filtered tracking error + RISE integral; returns tau_hat same shape as e_x."""
        e = e_x_dot + (self.kp / self.kv) * e_x
        if self.integral is None or self.integral.shape != e.shape:
            self.integral = torch.zeros_like(e)
        self.integral = self.integral + (e + self.beta * torch.sign(e)) * dt
        return self.alpha * e + self.integral


def self_check() -> None:
    r = RISEController()
    e = torch.tensor([0.1, -0.05])
    ed = torch.zeros(2)
    t0 = r.compute(e, ed, dt=0.05)
    t1 = r.compute(e, ed, dt=0.05)
    assert t0.shape == (2,)
    assert float(torch.norm(t1)) >= float(torch.norm(t0)) - 1e-6  # integral grows
    r.reset()
    assert r.integral is None
    print("rise_controller: OK")


if __name__ == "__main__":
    self_check()
