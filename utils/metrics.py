"""实验指标 — 对应论文第五章。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeMetrics:
    success_rate: float = 0.0
    collision_rate: float = 0.0
    communication_cost: float = 0.0
    episode_length: float = 0.0
    extra: dict = field(default_factory=dict)


def aggregate_success(episodes: list[bool]) -> float:
    return sum(episodes) / len(episodes) if episodes else 0.0
