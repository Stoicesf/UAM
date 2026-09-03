"""Failure injection library."""

from __future__ import annotations

from typing import Literal

import torch

FailureType = Literal[
    "node_kill", "comm_loss", "sensor_degradation", "byzantine", "energy_depletion"
]


class FailureInjector:
    FAILURE_TYPES = [
        "node_kill",
        "comm_loss",
        "sensor_degradation",
        "byzantine",
        "energy_depletion",
    ]

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.packet_loss = 0.0
        self.sensor_noise = 0.0
        self.byzantine = torch.zeros(n_agents, dtype=torch.bool)
        self.active: dict[str, torch.Tensor] = {}

    def inject(
        self,
        failure_type: FailureType,
        ratio: float,
        alive: torch.Tensor,
        batt: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        n = self.n_agents
        k = max(1, int(n * ratio))
        idx = torch.randperm(n, generator=generator)[:k]
        if failure_type == "node_kill":
            alive = alive.clone()
            alive[idx] = False
            self.active["node_kill"] = idx
        elif failure_type == "comm_loss":
            self.packet_loss = float(ratio)
            self.active["comm_loss"] = idx
        elif failure_type == "sensor_degradation":
            self.sensor_noise = 0.5 * ratio
            self.active["sensor_degradation"] = idx
        elif failure_type == "byzantine":
            self.byzantine = torch.zeros(n, dtype=torch.bool, device=alive.device)
            self.byzantine[idx] = True
            self.active["byzantine"] = idx
        elif failure_type == "energy_depletion":
            if batt is not None:
                batt = batt.clone()
                batt[idx] = 0.0
            alive = alive.clone()
            alive[idx] = False
            self.active["energy_depletion"] = idx
        return alive


def self_check() -> None:
    fi = FailureInjector(10)
    alive = torch.ones(10, dtype=torch.bool)
    alive2 = fi.inject("node_kill", 0.3, alive)
    assert int((~alive2).sum()) >= 3
    fi.inject("byzantine", 0.2, alive2)
    assert fi.byzantine.any()
    print("failure_injector: OK")


if __name__ == "__main__":
    self_check()
