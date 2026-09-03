"""Shared matplotlib helpers for demos / paper figures."""

from __future__ import annotations

from typing import Sequence

# Role colors: SCOUT / EXECUTOR / RELAY / GUARDIAN / LOGISTICS
ROLE_COLORS = {
    0: "#1a5fb4",  # blue SCOUT
    1: "#c01c28",  # red EXECUTOR
    2: "#2ec27e",  # green RELAY
    3: "#e66100",  # orange GUARDIAN
    4: "#813d9c",  # purple LOGISTICS
}

LEVEL_STYLE = {
    1: dict(linestyle="--", color="#888888", linewidth=0.5, alpha=0.45),
    2: dict(linestyle="-", color="#1a5fb4", linewidth=1.0, alpha=0.55),
    3: dict(linestyle="-", color="#c01c28", linewidth=2.0, alpha=0.7),
}

LEVEL_BYTES = {1: 8, 2: 16, 3: 64}


def role_color(role: int) -> str:
    return ROLE_COLORS.get(int(role) % 5, "#555555")


def colors_for_roles(roles: Sequence[int]) -> list[str]:
    return [role_color(r) for r in roles]
