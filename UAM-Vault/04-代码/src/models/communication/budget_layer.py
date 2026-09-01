"""Hard communication budget: keep top-k edges per agent.

Engineering constraint:
    maximize task return subject to sum_{j} 1[edge_ij] ≤ k_i
where k_i = ceil(budget_ratio * |N_i|)  OR  k_i = min(K_fixed, |N_i|).

T-RO Theorem 1: after projection, C(G_t) ≤ B_t (degree-wise: d_i ≤ k_i).
"""

from __future__ import annotations

import torch


def _scatter_topk(
    scores: torch.Tensor,
    adj_mask: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Keep top-k[i] scored neighbors per node. k: (B, N) int."""
    masked = scores * adj_mask.float()
    neg_inf = torch.finfo(masked.dtype).min
    masked = masked.masked_fill(adj_mask <= 0, neg_inf)

    k_max = int(k.max().item())
    k_max = max(1, min(k_max, masked.shape[-1]))
    top_vals, top_idx = torch.topk(masked, k=k_max, dim=-1)

    selected = torch.zeros_like(scores)
    arange_k = torch.arange(k_max, device=scores.device).view(1, 1, k_max)
    valid = arange_k < k.unsqueeze(-1)
    selected.scatter_(-1, top_idx, top_vals.masked_fill(~valid, 0.0))
    selected = selected * adj_mask.float()
    n = scores.shape[-1]
    eye = torch.eye(n, device=scores.device, dtype=scores.dtype).unsqueeze(0)
    return selected * (1.0 - eye)


def apply_topk_budget(
    scores: torch.Tensor,
    adj_mask: torch.Tensor,
    budget_ratio: float,
) -> torch.Tensor:
    """Keep top-k scored neighbors per node (vectorized where possible).

    Args:
        scores: (B, N, N) ranking scores (gates, quality, -distance, ...)
        adj_mask: (B, N, N) feasible edges (radius), diagonal already 0
        budget_ratio: in (0, 1]; 1.0 keeps all mask edges

    Returns:
        gated: (B, N, N) scores with non-selected entries zeroed
    """
    if budget_ratio is None or budget_ratio >= 1.0:
        return scores * adj_mask.float()

    if budget_ratio <= 0:
        return torch.zeros_like(scores)

    deg = adj_mask.float().sum(dim=-1).clamp(min=1.0)  # (B, N)
    k = (deg * float(budget_ratio)).ceil().long().clamp(min=1)
    return _scatter_topk(scores, adj_mask, k)


def apply_topk_fixed_k(
    scores: torch.Tensor,
    adj_mask: torch.Tensor,
    k_fixed: int,
) -> torch.Tensor:
    """Degree-budget projection: keep at most K edges per agent (Thm.~1 specialization)."""
    if k_fixed is None or k_fixed <= 0:
        return torch.zeros_like(scores)
    deg = adj_mask.float().sum(dim=-1)
    k = torch.minimum(torch.full_like(deg, float(k_fixed)), deg).long().clamp(min=0)
    # Nodes with deg=0 stay empty; clamp min=1 only when deg≥1
    k = torch.where(deg >= 1, k.clamp(min=1), k)
    return _scatter_topk(scores, adj_mask, k)


def hard_adjacency(gated: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """Binary adjacency after projection: 1[g_ij > threshold]."""
    return (gated > threshold).float()


def per_agent_degree(A: torch.Tensor) -> torch.Tensor:
    """Out-degree (B, N) or (N,) from binary / weighted adjacency."""
    if A.dim() == 2:
        return (A > 0).float().sum(dim=-1)
    return (A > 0).float().sum(dim=-1)


def degree_budget_caps(
    adj_mask: torch.Tensor,
    *,
    budget_ratio: float | None = None,
    k_fixed: int | None = None,
) -> torch.Tensor:
    """Per-agent allowed degree k_i (B, N)."""
    deg = adj_mask.float().sum(dim=-1)
    if k_fixed is not None:
        k = torch.minimum(torch.full_like(deg, float(k_fixed)), deg).long()
        return torch.where(deg >= 1, k.clamp(min=1), k)
    if budget_ratio is None or budget_ratio >= 1.0:
        return deg.long()
    if budget_ratio <= 0:
        return torch.zeros_like(deg, dtype=torch.long)
    return (deg.clamp(min=1.0) * float(budget_ratio)).ceil().long().clamp(min=1)


def degree_budget_violation(
    A: torch.Tensor,
    adj_mask: torch.Tensor,
    *,
    budget_ratio: float | None = None,
    k_fixed: int | None = None,
) -> dict[str, torch.Tensor | float]:
    """Thm.~1 feasibility diagnostics.

    V_B := max_i max(0, d_i - k_i)  (batch-reduced max), also report mean excess.
    Total edge budget B_tot = sum_i k_i; C = sum_i d_i.
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
        adj_mask = adj_mask.unsqueeze(0)
    d = per_agent_degree(A)  # (B, N)
    k = degree_budget_caps(adj_mask, budget_ratio=budget_ratio, k_fixed=k_fixed).float()
    excess = (d - k).clamp(min=0.0)
    C = d.sum(dim=-1)
    B_tot = k.sum(dim=-1)
    return {
        "degrees": d,
        "k_caps": k,
        "V_B": float(excess.max().item()),
        "V_B_mean": float(excess.mean().item()),
        "C": float(C.mean().item()),
        "B_t": float(B_tot.mean().item()),
        "V_B_total": float((C - B_tot).clamp(min=0.0).max().item()),
        "mean_degree": float(d.mean().item()),
        "rho": float(
            (C / (A.shape[-1] * (A.shape[-1] - 1) + 1e-8)).mean().item()
        ),
    }


def budget_edge_count(gated: torch.Tensor, threshold: float = 0.0) -> float:
    """Mean directed active edges after budget (score > threshold)."""
    if gated.dim() == 2:
        gated = gated.unsqueeze(0)
    return float((gated > threshold).float().sum(dim=(-2, -1)).mean())
