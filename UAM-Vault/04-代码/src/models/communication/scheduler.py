"""Budget / lambda scheduling for adaptive communication."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CommunicationScheduler:
    """Controls λ_c and optional hard budget on Σ g."""

    lambda_c: float = 1e-3
    lambda_warmup_fraction: float = 0.1
    budget_ratio: float | None = None  # e.g. 0.2 = keep 20% of mask edges
    hard_threshold: float = 0.5

    def lambda_at(self, step: int, total_steps: int) -> float:
        if total_steps <= 0 or self.lambda_warmup_fraction <= 0:
            return self.lambda_c
        warm = max(1, int(total_steps * self.lambda_warmup_fraction))
        return self.lambda_c * min(1.0, step / warm)

    def apply_budget(self, g: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        """Optionally keep top-k gates per row under budget_ratio * degree."""
        if self.budget_ratio is None or self.budget_ratio >= 1.0:
            return g
        # Per-node top-k among allowed neighbors
        b, n, _ = g.shape
        masked = g * adj_mask.float()
        deg = adj_mask.float().sum(dim=-1).clamp(min=1.0)
        k = (deg * self.budget_ratio).ceil().long()
        out = torch.zeros_like(masked)
        for bi in range(b):
            for i in range(n):
                ki = int(k[bi, i].item())
                if ki <= 0:
                    continue
                vals, idx = torch.topk(masked[bi, i], k=min(ki, n))
                out[bi, i, idx] = vals
        eye = torch.eye(n, device=g.device, dtype=g.dtype).unsqueeze(0)
        return out * (1.0 - eye)
