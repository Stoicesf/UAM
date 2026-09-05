"""Helpers for hierarchical training."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def save_checkpoint(module: nn.Module, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path)


def load_checkpoint(module: nn.Module, path: Path | str) -> None:
    module.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def load_config(path: Path | str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def light_scout_ratio(roles: torch.Tensor, uav_types: list[int]) -> float:
    light_idx = [i for i, t in enumerate(uav_types) if t == 2]
    if not light_idx:
        return 0.0
    return sum(1 for i in light_idx if int(roles[i]) == 0) / len(light_idx)


def pack_actions(dxdy: torch.Tensor, roles: torch.Tensor, n_roles: int) -> torch.Tensor:
    """dxdy + one_hot(roles)*2 so env conf > 0.4 accepts the role."""
    oh = torch.nn.functional.one_hot(roles.clamp(0, n_roles - 1), n_roles).float() * 2.0
    return torch.cat([dxdy, oh], dim=-1)


def eval_reward_formula(formula: str, coverage: float, collision: float) -> float:
    """Safe eval: only coverage/collision names and arithmetic."""
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in ("coverage", "collision"):
            raise ValueError(f"disallowed name in reward_formula: {node.id}")
        if isinstance(
            node, (ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.Import, ast.ImportFrom)
        ):
            raise ValueError(f"disallowed syntax in reward_formula: {type(node).__name__}")
    code = compile(tree, "<reward_formula>", "eval")
    return float(
        eval(code, {"__builtins__": {}}, {"coverage": float(coverage), "collision": float(collision)})
    )
