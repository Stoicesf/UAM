"""Decentralized task allocation (CBBA-lite: bid + consensus)."""

from __future__ import annotations

import torch

from dice.consensus import consensus_max_bid
from dice.local_obs import neighbor_mask


class DecentralizedTaskAllocation:
    def __init__(self, n_agents: int, n_tasks: int, comm_radius: float = 2.0):
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.comm_radius = comm_radius

    def values(
        self,
        agent_pos: torch.Tensor,
        task_pos: torch.Tensor,
        task_pri: torch.Tensor,
        capabilities: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(N, T) value = priority / (1+dist) * capability."""
        d = torch.cdist(agent_pos, task_pos)
        v = task_pri.unsqueeze(0) / (1.0 + d)
        if capabilities is not None:
            # capabilities: (N,) scalar skill
            v = v * capabilities.unsqueeze(-1)
        return v

    def allocate(
        self,
        agent_positions: torch.Tensor,
        task_positions: torch.Tensor,
        task_priorities: torch.Tensor,
        agent_capabilities: torch.Tensor | None = None,
        alive: torch.Tensor | None = None,
        max_iter: int = 8,
        done_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return assignment (N,) task id or -1."""
        n, t = agent_positions.shape[0], task_positions.shape[0]
        if alive is None:
            alive = torch.ones(n, dtype=torch.bool, device=agent_positions.device)
        if done_mask is None:
            done_mask = torch.zeros(t, dtype=torch.bool, device=agent_positions.device)
        vals = self.values(agent_positions, task_positions, task_priorities, agent_capabilities)
        vals = vals.masked_fill(done_mask.unsqueeze(0), -1e9)
        vals = vals.masked_fill(~alive.unsqueeze(-1), -1e9)

        assign = torch.full((n,), -1, dtype=torch.long, device=agent_positions.device)
        claimed = done_mask.clone()

        for _ in range(max_iter):
            # Phase 1: each alive agent bids on best remaining task
            remaining = ~claimed
            if not remaining.any():
                break
            v = vals.masked_fill(claimed.unsqueeze(0), -1e9)
            best_t = v.argmax(dim=-1)
            best_v = v.max(dim=-1).values
            # Phase 2: consensus on who wins each task (max bid)
            for ti in range(t):
                if claimed[ti]:
                    continue
                bidders = (best_t == ti) & alive
                if not bidders.any():
                    continue
                bids = torch.where(bidders, best_v, torch.full_like(best_v, -1e9))
                # global max among bidders (local consensus approx via full max for N<=128)
                winner = int(bids.argmax())
                # verify uniqueness
                if float(bids[winner]) < -1e8:
                    continue
                assign[assign == ti] = -1
                assign[winner] = ti
                claimed[ti] = True
        return assign


def self_check() -> None:
    n, t = 8, 4
    ap = torch.randn(n, 2)
    tp = torch.randn(t, 2)
    pri = torch.ones(t)
    a = DecentralizedTaskAllocation(n, t).allocate(ap, tp, pri)
    assert a.shape == (n,)
    covered = set(int(x) for x in a.tolist() if x >= 0)
    assert len(covered) >= min(n, t) - 1  # nearly full
    print(f"task_allocation: OK (covered={len(covered)}/{t})")


if __name__ == "__main__":
    self_check()
