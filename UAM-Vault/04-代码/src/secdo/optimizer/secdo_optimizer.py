"""SECDO v2 optimizer — PI-conditioned mixed anticipatory projection.

c_mix = α ĉ + (1-α) c_now,  α = 1/(1+PI²),  PI = δ̂/(χ̂+ε)
Training: pass detach=True so ĉ does not receive grad through Π.
"""

from __future__ import annotations

import torch

from secdo.optimizer.anticipatory_projection import project_sum_budget
from secdo.optimizer.projected_gradient import gradient_step
from secdo.optimizer.reactive_projection import reactive_project


def predictability_index(
    delta_hat: torch.Tensor,
    chi_hat: torch.Tensor,
    eps: float = 1e-8,
    chi_floor: float = 1e-2,
) -> torch.Tensor:
    """PI = δ / χ with χ floored to avoid numerical blow-up when drift≈0."""
    return delta_hat / (chi_hat.abs().clamp(min=chi_floor) + eps)


def anticipation_weight(pi: torch.Tensor) -> torch.Tensor:
    """α = 1 / (1 + PI²) ∈ (0,1]; PI≫1 ⇒ α→0 (reactive)."""
    return 1.0 / (1.0 + pi.pow(2))


def mix_budget(
    c_hat: torch.Tensor,
    c_now: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    a = alpha.view_as(c_hat) if alpha.shape != c_hat.shape else alpha
    return a * c_hat + (1.0 - a) * c_now


class SECDOOptimizer:
    def __init__(self, eta: float = 0.25, pi_eps: float = 1e-8):
        self.eta = eta
        self.pi_eps = pi_eps

    def step(
        self,
        x: torch.Tensor,
        grad: torch.Tensor | None,
        c_hat: torch.Tensor,
        c_now: torch.Tensor,
        delta_hat: torch.Tensor | None = None,
        chi_hat: torch.Tensor | None = None,
        *,
        pref: torch.Tensor | None = None,
        detach_budget: bool = True,
        fixed_alpha: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_next, alpha, PI
        """
        if grad is None:
            if pref is None:
                raise ValueError("need grad or pref for F(x)=0.5||x-pref||²")
            y = gradient_step(x, pref, self.eta)
        else:
            y = x - self.eta * grad

        c_hat_use = c_hat.detach() if detach_budget else c_hat

        if fixed_alpha is not None:
            alpha = torch.full_like(c_now, float(fixed_alpha))
            pi = predictability_index(
                delta_hat if delta_hat is not None else (c_hat_use - c_now).abs(),
                chi_hat if chi_hat is not None else torch.ones_like(c_now),
                self.pi_eps,
            )
        else:
            if delta_hat is None:
                delta_hat = (c_hat_use - c_now).abs()
            if chi_hat is None:
                chi_hat = torch.ones_like(c_now) * self.pi_eps
            pi = predictability_index(delta_hat, chi_hat, self.pi_eps)
            alpha = anticipation_weight(pi)

        c_mix = mix_budget(c_hat_use, c_now, alpha)
        x_next = project_sum_budget(y, c_mix)
        return x_next, alpha, pi

    def reactive_step(self, x: torch.Tensor, pref: torch.Tensor, c_now: torch.Tensor) -> torch.Tensor:
        y = gradient_step(x, pref, self.eta)
        return reactive_project(y, c_now)

    def oracle_step(self, x: torch.Tensor, pref: torch.Tensor, c_next: torch.Tensor) -> torch.Tensor:
        y = gradient_step(x, pref, self.eta)
        return project_sum_budget(y, c_next.detach())


SecdoOptimizer = SECDOOptimizer
