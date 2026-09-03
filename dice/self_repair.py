"""Self-repair: detect failure → reallocate tasks/roles → reform."""

from __future__ import annotations

import torch

from dice.role_negotiation import RoleNegotiation
from dice.task_realloc import TaskReallocator
from dice.task_allocation import DecentralizedTaskAllocation


class SelfRepair:
    def __init__(self, n_agents: int, n_tasks: int, n_roles: int = 3, comm_radius: float = 2.0):
        self.allocator = DecentralizedTaskAllocation(n_agents, n_tasks, comm_radius)
        self.reallocator = TaskReallocator(self.allocator)
        self.roles = RoleNegotiation(n_roles, comm_radius)
        self.last_heartbeat = torch.zeros(n_agents)

    def detect_failure(self, alive: torch.Tensor, batt: torch.Tensor, step: int) -> torch.Tensor:
        """Mark failed if not alive or battery empty."""
        failed = (~alive) | (batt <= 1e-6)
        self.last_heartbeat = torch.where(alive, torch.full_like(self.last_heartbeat, float(step)), self.last_heartbeat)
        return failed

    def trigger_repair(
        self,
        pos: torch.Tensor,
        tasks: torch.Tensor,
        roles: torch.Tensor,
        alive: torch.Tensor,
        done_mask: torch.Tensor,
        assignment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assign = self.reallocator.reallocate(
            pos, tasks[:, :2], tasks[:, 2], alive, done_mask, assignment
        )
        n, r = pos.shape[0], self.roles.n_roles
        demand = torch.ones(n, r, device=pos.device) * 0.5
        cap = torch.ones(n, r, device=pos.device)
        new_roles = self.roles.negotiate(pos, cap, roles, demand, alive)
        return assign, new_roles


def self_check() -> None:
    sr = SelfRepair(6, 3)
    pos = torch.randn(6, 2)
    tasks = torch.randn(3, 4)
    tasks[:, 2] = 1
    roles = torch.zeros(6, dtype=torch.long)
    alive = torch.ones(6, dtype=torch.bool)
    alive[1] = False
    done = torch.zeros(3, dtype=torch.bool)
    asn = torch.full((6,), -1, dtype=torch.long)
    a2, r2 = sr.trigger_repair(pos, tasks, roles, alive, done, asn)
    assert a2.shape == (6,)
    print("self_repair: OK")


if __name__ == "__main__":
    self_check()
