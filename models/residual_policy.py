"""Residual Guidance Policy — a = a_base + β·Δa(Φ).

Decouples guidance from reward shaping: Φ acts as policy prior, not surrogate objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP


class ResidualGuidanceActor(nn.Module):
    """NavRL base policy on obs + guidance correction on Φ.

    a_i = π_θ(o_i) + β · f_φ(Φ_i)
    β decays over training (updated externally via .beta buffer).
    """

    def __init__(
        self,
        obs_dim: int,
        phi_dim: int,
        action_dim: int,
        n_agents: int,
        hidden_dim: int = 128,
        share_params: bool = True,
        device: torch.device | str = "cpu",
        beta_init: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.base_mlp = MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=n_agents,
            centralised=False,
            share_params=share_params,
            device=device,
            depth=2,
            num_cells=hidden_dim,
            activation_class=nn.Tanh,
        )
        self.delta_net = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.extractor = NormalParamExtractor()
        self.register_buffer("beta", torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, obs: torch.Tensor, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loc_base, scale = self.extractor(self.base_mlp(obs))
        delta = self.delta_net(phi)
        loc = loc_base + self.beta * delta
        return loc, scale

    def set_beta(self, value: float) -> None:
        self.beta.fill_(value)
