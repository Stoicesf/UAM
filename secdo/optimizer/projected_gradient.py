"""Unconstrained gradient step y = x - η ∇F(x)."""

from __future__ import annotations

import torch


def gradient_step(x: torch.Tensor, pref: torch.Tensor, eta: float) -> torch.Tensor:
    """F(x)=0.5‖x-pref‖² ⇒ ∇F = x - pref."""
    return x - eta * (x - pref)
