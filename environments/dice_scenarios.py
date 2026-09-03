"""Scenario presets for DICE experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ScenarioSpec:
    name: str
    n_agents: int = 16
    n_tasks: int = 5
    task_type: str = "point_coverage"
    max_steps: int = 200
    comm_radius: float = 2.0
    failure_ratio: float = 0.0
    track_speed: float = 1.0


SCENARIOS: dict[str, ScenarioSpec] = {
    "search": ScenarioSpec("search", task_type="point_coverage", n_tasks=5),
    "tracking": ScenarioSpec("tracking", task_type="dynamic_tracking", n_tasks=4, track_speed=1.0),
    "adversarial": ScenarioSpec("adversarial", task_type="point_coverage", failure_ratio=0.3),
    "area": ScenarioSpec("area", task_type="area_coverage", n_tasks=6),
}

SCALE_NS = [4, 8, 16, 32, 64, 128]


def kwargs_for(spec: ScenarioSpec, n_agents: int | None = None) -> dict:
    d = asdict(spec)
    d.pop("name")
    d.pop("failure_ratio")
    d.pop("track_speed")
    if n_agents is not None:
        d["n_agents"] = n_agents
    return d
