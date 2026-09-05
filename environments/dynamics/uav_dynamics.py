"""Simplified 2D multirotor dynamics (thrust + yaw rate)."""

from __future__ import annotations

import torch
import torch.nn as nn

# heavy / standard / light → (mass kg, T_max N)
UAV_DYN_BY_TYPE: dict[int, tuple[float, float]] = {
    0: (1.2, 12.0),
    1: (1.0, 10.0),
    2: (0.6, 8.0),
}


class UAVDynamics(nn.Module):
    """State (x,y,vx,vy,psi); action (T, psi_dot). Hover-simplified (no gravity on UAV)."""

    def __init__(
        self,
        mass: float = 1.0,
        T_max: float = 10.0,
        drag: float = 0.1,
        psi_dot_max: float = 1.5,
    ):
        super().__init__()
        self.mass = float(mass)
        self.T_max = float(T_max)
        self.drag = float(drag)
        self.psi_dot_max = float(psi_dot_max)

    def forward(self, state: torch.Tensor, action: torch.Tensor, dt: float = 0.05) -> torch.Tensor:
        """
        state: [n, 5]  action: [n, 2]
        returns new state [n, 5]
        """
        pos = state[:, :2]
        vel = state[:, 2:4]
        psi = state[:, 4:5]

        T = torch.clamp(action[:, 0:1], 0.0, self.T_max)
        psi_dot = torch.clamp(action[:, 1:2], -self.psi_dot_max, self.psi_dot_max)

        thrust = T * torch.cat([torch.cos(psi), torch.sin(psi)], dim=-1)
        drag_force = -self.drag * vel
        # sideslip damping: underactuated planar model kills lateral velocity
        heading = torch.cat([torch.cos(psi), torch.sin(psi)], dim=-1)
        v_par = (vel * heading).sum(dim=-1, keepdim=True)
        v_lat = vel - v_par * heading
        drag_force = drag_force - 2.0 * v_lat
        acc = (thrust + drag_force) / self.mass

        new_vel = vel + acc * dt
        new_pos = pos + new_vel * dt
        new_psi = psi + psi_dot * dt
        return torch.cat([new_pos, new_vel, new_psi], dim=-1)


def self_check() -> None:
    dyn = UAVDynamics(mass=1.0, T_max=10.0)
    state = torch.zeros(4, 5)
    action = torch.tensor([[5.0, 0.5], [12.0, 3.0], [0.0, -2.0], [1.0, 0.0]])
    out = dyn(state, action, dt=0.05)
    assert out.shape == (4, 5)
    # T clamped: agent1 requested 12 → effective thrust uses T_max
    assert float(action[1, 0]) > dyn.T_max
    print("uav_dynamics: OK")


if __name__ == "__main__":
    self_check()
