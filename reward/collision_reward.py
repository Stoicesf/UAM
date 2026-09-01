"""碰撞惩罚。"""

from __future__ import annotations

import torch


def collision_penalty(collided: torch.Tensor, penalty: float = -1.0) -> torch.Tensor:
    return collided.float() * penalty
