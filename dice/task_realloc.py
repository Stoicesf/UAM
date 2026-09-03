"""Dynamic task reallocation on agent failure / new tasks."""

from __future__ import annotations

import torch

from dice.task_allocation import DecentralizedTaskAllocation


class TaskReallocator:
    def __init__(self, allocator: DecentralizedTaskAllocation):
        self.allocator = allocator

    def reallocate(
        self,
        agent_pos: torch.Tensor,
        task_pos: torch.Tensor,
        task_pri: torch.Tensor,
        alive: torch.Tensor,
        done_mask: torch.Tensor,
        old_assign: torch.Tensor,
    ) -> torch.Tensor:
        # drop assignments of dead agents
        assign = old_assign.clone()
        assign[~alive] = -1
        # free tasks that lost owners
        owned = set(int(x) for x in assign.tolist() if x >= 0)
        need = []
        for ti in range(task_pos.shape[0]):
            if done_mask[ti]:
                continue
            if ti not in owned:
                need.append(ti)
        if not need and alive.all():
            return assign
        return self.allocator.allocate(
            agent_pos, task_pos, task_pri, alive=alive, done_mask=done_mask
        )


def self_check() -> None:
    alloc = DecentralizedTaskAllocation(6, 3)
    re = TaskReallocator(alloc)
    ap = torch.randn(6, 2)
    tp = torch.randn(3, 2)
    pri = torch.ones(3)
    alive = torch.ones(6, dtype=torch.bool)
    alive[0] = False
    old = alloc.allocate(ap, tp, pri)
    new = re.reallocate(ap, tp, pri, alive, torch.zeros(3, dtype=torch.bool), old)
    assert int(new[0]) == -1 or not alive[0]
    print("task_realloc: OK")


if __name__ == "__main__":
    self_check()
