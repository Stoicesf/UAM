"""Role emergence reward shaping."""

from __future__ import annotations

import torch


class RoleReward:
    def __init__(self, n_roles: int = 3):
        self.n_roles = n_roles

    def diversity_reward(self, roles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-agent: fraction of distinct neighbor roles."""
        n = roles.shape[0]
        out = torch.zeros(n, device=roles.device)
        for i in range(n):
            neigh = roles[mask[i]]
            if neigh.numel() == 0:
                out[i] = 0.0
                continue
            uniq = torch.unique(torch.cat([neigh, roles[i : i + 1]])).numel()
            out[i] = uniq / float(self.n_roles)
        return out

    def complementarity_reward(self, roles: torch.Tensor) -> torch.Tensor:
        """Global: encourage covering all role types; broadcast to agents."""
        counts = torch.bincount(roles, minlength=self.n_roles).float()
        # entropy of role distribution
        p = counts / counts.sum().clamp(min=1)
        ent = -(p * (p + 1e-8).log()).sum()
        return torch.full((roles.shape[0],), float(ent), device=roles.device)

    def stability_reward(self, roles: torch.Tensor, prev_roles: torch.Tensor) -> torch.Tensor:
        return (roles == prev_roles).float()

    def capability_match_bonus(
        self,
        roles: torch.Tensor,
        uav_types: list[int] | torch.Tensor,
        uav_types_table: dict | None = None,
    ) -> torch.Tensor:
        """Per-agent bonus: light↔SCOUT/EXECUTOR, heavy↔RELAY/GUARDIAN."""
        from environments.dice_vmas_env import UAV_TYPES

        table = uav_types_table or UAV_TYPES
        # role → {type_name: score}
        match_matrix = {
            0: {"light": 1.0, "standard": 0.5, "heavy": 0.0},
            1: {"light": 0.8, "standard": 0.8, "heavy": 0.2},
            2: {"light": 0.0, "standard": 0.5, "heavy": 1.0},
            3: {"light": 0.0, "standard": 0.3, "heavy": 1.0},
            4: {"light": 0.2, "standard": 0.8, "heavy": 0.8},
        }
        if isinstance(uav_types, torch.Tensor):
            types = uav_types.detach().cpu().tolist()
        else:
            types = list(uav_types)
        out = torch.zeros(roles.shape[0], device=roles.device)
        for i in range(roles.shape[0]):
            role = int(roles[i].item())
            name = table[int(types[i])]["name"]
            out[i] = match_matrix.get(role, {}).get(name, 0.5) * 0.2
        return out

    def compute(
        self,
        roles: torch.Tensor,
        prev_roles: torch.Tensor,
        mask: torch.Tensor,
        *,
        uav_types: list[int] | torch.Tensor | None = None,
        match_weight: float = 1.0,
    ) -> torch.Tensor:
        r = (
            self.diversity_reward(roles, mask)
            + 0.5 * self.complementarity_reward(roles)
            + 0.3 * self.stability_reward(roles, prev_roles)
        )
        if uav_types is not None and match_weight > 0:
            r = r + float(match_weight) * self.capability_match_bonus(roles, uav_types)
        return r


def role_entropy(roles: torch.Tensor, n_roles: int = 3) -> float:
    counts = torch.bincount(roles.cpu(), minlength=n_roles).float()
    p = counts / counts.sum().clamp(min=1)
    return float(-(p * (p + 1e-8).log()).sum())


def self_check() -> None:
    roles = torch.tensor([0, 1, 2, 0, 1, 2])
    prev = roles.clone()
    mask = torch.ones(6, 6).bool()
    mask.fill_diagonal_(False)
    rr = RoleReward(3)
    r = rr.compute(roles, prev, mask)
    assert r.shape == (6,)
    # light SCOUT > heavy SCOUT
    bonus = rr.capability_match_bonus(torch.tensor([0, 0]), [2, 0])
    assert float(bonus[0]) > float(bonus[1])
    r2 = rr.compute(roles, prev, mask, uav_types=[2, 1, 0, 2, 1, 0])
    assert r2.shape == (6,)
    ent = role_entropy(roles, 3)
    assert ent > 1.0  # near-uniform 3-class
    print(f"role_reward: OK (entropy={ent:.3f}, match={float(bonus[0]):.2f})")


if __name__ == "__main__":
    self_check()
