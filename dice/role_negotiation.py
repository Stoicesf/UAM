"""Decentralized role negotiation via local bidding."""

from __future__ import annotations

import torch

from dice.local_obs import neighbor_mask


class RoleNegotiation:
    def __init__(self, n_roles: int = 3, comm_radius: float = 2.0):
        self.n_roles = n_roles
        self.comm_radius = comm_radius

    def _fitness(
        self,
        capabilities: torch.Tensor,
        task_demand: torch.Tensor,
    ) -> torch.Tensor:
        """(n_agents, n_roles) fitness."""
        # capabilities: (N, R) preferred roles; demand: (N, R) local need
        return capabilities * (0.5 + task_demand)

    def negotiate(
        self,
        positions: torch.Tensor,
        capabilities: torch.Tensor,
        current_roles: torch.Tensor,
        task_demands: torch.Tensor,
        alive: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = positions.shape[0]
        if alive is None:
            alive = torch.ones(n, dtype=torch.bool, device=positions.device)
        mask = neighbor_mask(positions, self.comm_radius, alive)
        fit = self._fitness(capabilities, task_demands)
        new_roles = current_roles.clone()
        for i in range(n):
            if not alive[i]:
                continue
            my = fit[i]
            # conflict: if a neighbor already claims top role with higher fitness, pick next
            order = torch.argsort(my, descending=True)
            chosen = int(order[0])
            for r in order.tolist():
                conflict = False
                for j in torch.where(mask[i])[0].tolist():
                    if int(new_roles[j]) == r and float(fit[j, r]) > float(my[r]):
                        conflict = True
                        break
                if not conflict:
                    chosen = r
                    break
            new_roles[i] = chosen
        return new_roles


def self_check() -> None:
    n, r = 8, 3
    pos = torch.randn(n, 2)
    cap = torch.rand(n, r)
    roles = torch.zeros(n, dtype=torch.long)
    demand = torch.rand(n, r)
    out = RoleNegotiation(r).negotiate(pos, cap, roles, demand)
    assert out.shape == (n,)
    assert int(out.max()) < r
    print("role_negotiation: OK")


if __name__ == "__main__":
    self_check()
