"""Reactive baseline: Π_{B(c_t)}(y) — no prediction."""

from __future__ import annotations

import torch

from secdo.optimizer.anticipatory_projection import project_sum_budget


def reactive_project(y: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
    return project_sum_budget(y, c_t)
