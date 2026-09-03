"""Action projection safety shield — no cvxpy.

# ponytail: analytic repulsion scaling, not formal CBF-QP; upgrade if collisions remain.
"""

from __future__ import annotations

import torch


def project_action(
    u_nom: torch.Tensor,
    pos: torch.Tensor,
    *,
    boundary: float = 5.0,
    min_dist: float = 0.5,
    return_correction: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Minimal correction so pos+u stays inside bounds and away from neighbors."""
    u = u_nom.clone()
    proposed = pos + u

    # boundary: zero outward component past wall
    over = proposed.abs() > boundary
    u = torch.where(over & (proposed * u > 0), torch.zeros_like(u), u)

    # pairwise: if too close after step, push apart along separation
    n = pos.shape[0]
    proposed = pos + u
    for i in range(n):
        for j in range(i + 1, n):
            d = proposed[i] - proposed[j]
            dist = float(d.norm())
            if dist < min_dist and dist > 1e-8:
                direction = d / dist
                deficit = (min_dist - dist) * 0.5
                u[i] = u[i] + direction * deficit
                u[j] = u[j] - direction * deficit
            elif dist <= 1e-8:
                u[i, 0] = u[i, 0] + 0.1
                u[j, 0] = u[j, 0] - 0.1
    u = u.clamp(-1.0, 1.0)
    if return_correction:
        return u, (u - u_nom)
    return u


def self_check() -> None:
    pos = torch.tensor([[0.0, 0.0], [0.2, 0.0]], dtype=torch.float32)
    u = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    u2 = project_action(u, pos, min_dist=0.5)
    sep = ((pos + u2)[0] - (pos + u2)[1]).norm()
    assert float(sep) >= 0.49
    hard = (pos + u)  # would stay at 0.2
    hard_delta = float((u2 - u).norm())
    assert hard_delta > 0
    print(f"safety_shield_projection: OK (sep={float(sep):.3f}, ||du||={hard_delta:.3f})")


if __name__ == "__main__":
    self_check()
