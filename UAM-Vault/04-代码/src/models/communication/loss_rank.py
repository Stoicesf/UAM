"""Pairwise ranking loss for action-causal utility.

Encourages U to preserve order of U*: neighbors with higher
action influence should receive higher predicted utility.

Default (v2d): logistic Bradley–Terry
    L_rank = −log σ(U_hi − U_lo)  when U*_hi > U*_lo

Optional: hinge ranking (legacy v2c).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def utility_ranking_loss(
    u_pred: torch.Tensor,
    u_star: torch.Tensor,
    pair_mask: torch.Tensor | None = None,
    margin: float = 1.0,
    max_pairs: int = 256,
    mode: str = "hinge",
) -> torch.Tensor:
    """Pairwise ranking on flattened valid edges.

    Args:
        u_pred: (B, N, N) predicted utilities
        u_star: (B, N, N) detached targets
        pair_mask: optional (B, N, N) valid edges
        margin: hinge margin (mode='hinge' only)
        max_pairs: subsample for speed
        mode: 'logistic' | 'hinge'  (default hinge = v2c / v2d-A′)

    Returns:
        scalar ranking loss
    """
    u_star = u_star.detach()
    flat_p = u_pred.reshape(-1)
    flat_t = u_star.reshape(-1)
    if pair_mask is not None:
        m = pair_mask.reshape(-1) > 0
        flat_p = flat_p[m]
        flat_t = flat_t[m]
    n = flat_p.numel()
    if n < 2:
        return u_pred.new_zeros(())

    n_pairs = min(max_pairs, n * (n - 1) // 2)
    i = torch.randint(0, n, (n_pairs,), device=u_pred.device)
    j = torch.randint(0, n, (n_pairs,), device=u_pred.device)
    same = i == j
    if same.any():
        j = torch.where(same, (j + 1) % n, j)

    t_i, t_j = flat_t[i], flat_t[j]
    p_i, p_j = flat_p[i], flat_p[j]

    active = (t_i - t_j).abs() > 1e-6
    if not active.any():
        return u_pred.new_zeros(())

    # Order by U*: want pred(hi) > pred(lo)
    hi_first = t_i > t_j
    p_hi = torch.where(hi_first, p_i, p_j)
    p_lo = torch.where(hi_first, p_j, p_i)
    diff = p_hi - p_lo

    if mode == "logistic":
        # −log σ(U_hi − U_lo) — can distort absolute U scale
        loss = -F.logsigmoid(diff)
    else:
        # hinge (default): max(0, m − (U_hi − U_lo))
        loss = F.relu(margin - diff)

    return loss[active].mean()
