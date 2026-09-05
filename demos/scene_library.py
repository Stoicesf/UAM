"""Preset scenes for interactive / recorded demos."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DemoScene:
    name: str
    n_tasks: int = 5
    task_type: str = "point_coverage"
    obstacle_density: float = 0.0
    failure_ratio: float = 0.0
    comm_loss: float = 0.0
    target_speed: float = 1.0
    max_steps: int = 300
    snr_db: float = 10.0
    bandwidth_scale: float = 0.5
    kill_at_frac: float = 0.33
    # sim-to-real
    sensor_noise: float = 0.0
    drop_rate: float = 0.0
    comm_delay: int = 0
    max_acc: float = float("inf")
    max_vel: float = float("inf")
    max_turn_deg: float = 180.0
    # heterogeneous swarm
    heterogeneous: bool = False
    hetero_ratio: tuple[float, float, float] = (0.3, 0.4, 0.3)
    # pursuit
    evader_speed: float = 1.2
    capture_radius: float = 2.0
    default_n_agents: int | None = None
    evader_policy: str = "scripted"  # scripted | rl
    evader_ckpt: str = ""


SCENES: dict[str, DemoScene] = {
    "search": DemoScene("search", n_tasks=8, task_type="point_coverage", max_steps=300),
    "tracking": DemoScene(
        "tracking", n_tasks=3, task_type="dynamic_tracking", target_speed=1.0, max_steps=400
    ),
    "adversarial": DemoScene(
        "adversarial", n_tasks=5, task_type="point_coverage", failure_ratio=0.2, comm_loss=0.1, max_steps=500
    ),
    "mixed": DemoScene(
        "mixed",
        n_tasks=6,
        task_type="point_coverage",
        obstacle_density=0.1,
        failure_ratio=0.1,
        max_steps=400,
    ),
    "pursuit": DemoScene(
        "pursuit",
        n_tasks=1,
        task_type="pursuit",
        evader_speed=1.2,
        capture_radius=2.0,
        max_steps=400,
        default_n_agents=12,
        evader_policy="scripted",
    ),
    "adversarial_pursuit": DemoScene(
        "adversarial_pursuit",
        n_tasks=1,
        task_type="pursuit",
        evader_speed=1.4,
        capture_radius=2.0,
        max_steps=400,
        default_n_agents=12,
        evader_policy="rl",
    ),
    "transport": DemoScene(
        "transport",
        n_tasks=1,
        task_type="transport",
        max_steps=400,
        heterogeneous=True,
        default_n_agents=16,
    ),
}


def get_scene(name: str) -> DemoScene:
    key = name.strip().lower()
    if key not in SCENES:
        raise KeyError(f"Unknown scene '{name}'. Choose from: {list(SCENES)}")
    return SCENES[key]


def env_kwargs(scene: DemoScene, n_agents: int) -> dict:
    tt = scene.task_type if scene.task_type != "mixed" else "point_coverage"
    return {
        "n_agents": n_agents,
        "n_tasks": scene.n_tasks,
        "task_type": tt,
        "max_steps": scene.max_steps,
        "sensor_noise": scene.sensor_noise,
        "vel_noise": 0.02 if scene.sensor_noise > 0 else 0.0,
        "drop_rate": scene.drop_rate,
        "max_acc": scene.max_acc,
        "max_vel": scene.max_vel,
        "max_turn_deg": scene.max_turn_deg,
        "evader_speed": scene.evader_speed,
        "capture_radius": scene.capture_radius,
        "evader_policy": scene.evader_policy,
        "evader_ckpt": scene.evader_ckpt,
        "heterogeneous": scene.heterogeneous,
        "hetero_ratio": tuple(scene.hetero_ratio),
    }


def as_dict(scene: DemoScene) -> dict:
    return asdict(scene)
