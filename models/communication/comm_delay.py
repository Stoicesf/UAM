"""Ring buffer delaying semantic send/level decisions (sim-to-real comm delay)."""

from __future__ import annotations

from collections import deque

import torch


class CommDelayBuffer:
    """Delay (send, levels) by ``delay_steps`` ticks. delay_steps=0 → passthrough."""

    def __init__(self, delay_steps: int = 0):
        self.delay_steps = max(0, int(delay_steps))
        self._q: deque[tuple[torch.Tensor, torch.Tensor]] = deque(maxlen=self.delay_steps + 1)

    def reset(self) -> None:
        self._q.clear()

    def push(self, send: torch.Tensor, levels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.delay_steps <= 0:
            return send, levels
        self._q.append((send.detach().clone(), levels.detach().clone()))
        if len(self._q) <= self.delay_steps:
            # not enough history yet → no send
            z_send = torch.zeros_like(send, dtype=torch.bool)
            z_lvl = torch.zeros_like(levels)
            return z_send, z_lvl
        return self._q[0]
