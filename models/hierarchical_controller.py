"""Two-layer controller: DICE high-level + AC-DSGF-style low-level motion."""

from __future__ import annotations

import torch
import torch.nn as nn

from dice.local_rules import LocalRules
from dice.role_negotiation import RoleNegotiation
from dice.safety_shield import SafetyShield
from dice.task_allocation import DecentralizedTaskAllocation
from dice.task_realloc import TaskReallocator


class HierarchicalController(nn.Module):
    """High: task+role; Low: local rules (+ optional AC-DSGF hook)."""

    def __init__(
        self,
        n_agents: int,
        n_tasks: int,
        n_roles: int = 3,
        comm_radius: float = 2.0,
        ac_dsgf: nn.Module | None = None,
    ):
        super().__init__()
        self.n_roles = n_roles
        self.allocator = DecentralizedTaskAllocation(n_agents, n_tasks, comm_radius)
        self.reallocator = TaskReallocator(self.allocator)
        self.roles = RoleNegotiation(n_roles, comm_radius)
        self.rules = LocalRules(comm_radius)
        self.ac_dsgf = ac_dsgf  # optional; if set, used for guidance residual
        self.comm_radius = comm_radius

    @torch.no_grad()
    def step(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        tasks: torch.Tensor,
        roles: torch.Tensor,
        alive: torch.Tensor,
        done_mask: torch.Tensor,
        assignment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns dxdy actions (N,2), new_roles, new_assignment, goals."""
        task_pos = tasks[:, :2]
        pri = tasks[:, 2]
        assignment = self.reallocator.reallocate(
            pos, task_pos, pri, alive, done_mask, assignment
        )
        # goals from assignment
        goals = pos.clone()
        for i, ti in enumerate(assignment.tolist()):
            if ti >= 0 and alive[i]:
                goals[i] = task_pos[ti]
        # role demands from local unfinished tasks
        n = pos.shape[0]
        demand = torch.zeros(n, self.n_roles, device=pos.device)
        for i in range(n):
            if not alive[i]:
                continue
            d = (task_pos - pos[i]).norm(dim=-1)
            near = (d < self.comm_radius) & (~done_mask)
            if near.any():
                demand[i, 0] = 0.5  # scout
                demand[i, 1] = 1.0  # executor
                demand[i, 2] = 0.3  # relay
        cap = torch.ones(n, self.n_roles, device=pos.device)
        new_roles = self.roles.negotiate(pos, cap, roles, demand, alive)
        acc = self.rules(pos, vel, goals)
        if self.ac_dsgf is not None:
            # optional residual from AC-DSGF phi[..., :2]
            try:
                obs = torch.cat([pos, vel], dim=-1)
                phi, _, _ = self.ac_dsgf(obs)
                acc = acc + 0.3 * phi[..., :2].squeeze(0)
            except Exception:
                pass
        dxdy = acc.clamp(-1, 1)
        return dxdy, new_roles, assignment, goals


def self_check() -> None:
    ctl = HierarchicalController(8, 4)
    pos = torch.randn(8, 2)
    vel = torch.zeros(8, 2)
    tasks = torch.randn(4, 4)
    tasks[:, 2] = 1
    roles = torch.zeros(8, dtype=torch.long)
    alive = torch.ones(8, dtype=torch.bool)
    done = torch.zeros(4, dtype=torch.bool)
    assign = torch.full((8,), -1, dtype=torch.long)
    a, r, asn, g = ctl.step(pos, vel, tasks, roles, alive, done, assign)
    assert a.shape == (8, 2)
    print("hierarchical_controller: OK")


if __name__ == "__main__":
    self_check()
