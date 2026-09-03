"""Task generation for DICE scenarios."""

from __future__ import annotations

import torch


class TaskGenerator:
    TASK_TYPES = ["point_coverage", "area_coverage", "dynamic_tracking"]

    def __init__(
        self,
        n_tasks: int,
        task_type: str = "point_coverage",
        boundary: float = 5.0,
        device: str | torch.device = "cpu",
    ):
        self.n_tasks = n_tasks
        self.task_type = task_type
        self.boundary = boundary
        self.device = torch.device(device)

    def generate(self, generator: torch.Generator | None = None) -> torch.Tensor:
        g = generator
        xy = (torch.rand(self.n_tasks, 2, generator=g, device=self.device) * 2 - 1) * (
            self.boundary * 0.8
        )
        if self.task_type == "area_coverage":
            xy = xy.abs() * 0.5 + self.boundary * 0.15
        pri = torch.rand(self.n_tasks, 1, generator=g, device=self.device) * 0.5 + 0.5
        tw = torch.full((self.n_tasks, 1), 200.0, device=self.device)
        return torch.cat([xy, pri, tw], dim=-1)

    def update(self, tasks: torch.Tensor, completed: torch.Tensor) -> torch.Tensor:
        """Replace completed tasks with new samples."""
        out = tasks.clone()
        for i in range(tasks.shape[0]):
            if completed[i]:
                out[i] = self.generate()[0]
        return out


def self_check() -> None:
    g = TaskGenerator(4, "point_coverage")
    t = g.generate()
    assert t.shape == (4, 4)
    print("task_models: OK")


if __name__ == "__main__":
    self_check()
