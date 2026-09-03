"""Lightweight single-agent PPO for the pursuit evader (no torchrl)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal


class EvaderActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.zeros(2))
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        h = self.body(obs)
        mu = torch.tanh(self.mu(h))
        std = self.log_std.exp().clamp(1e-3, 1.0)
        return Normal(mu, std), self.v(h).squeeze(-1)


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    clip: float = 0.2
    epochs: int = 4
    ent_coef: float = 0.01


class EvaderPPO:
    def __init__(self, obs_dim: int, cfg: PPOConfig | None = None):
        self.cfg = cfg or PPOConfig()
        self.net = EvaderActorCritic(obs_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.buf: list[dict] = []

    def obs_dim_for(n_uav: int) -> int:
        # self pos2+vel2 + relative uav pos (n*2)
        return 4 + n_uav * 2

    @staticmethod
    def build_obs(evader_xy: torch.Tensor, evader_vel: torch.Tensor, uav_pos: torch.Tensor) -> torch.Tensor:
        rel = (uav_pos - evader_xy).reshape(-1)
        return torch.cat([evader_xy, evader_vel, rel], dim=0)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.net(obs.unsqueeze(0) if obs.dim() == 1 else obs)
        action = dist.mean if deterministic else dist.sample()
        action = action.squeeze(0).clamp(-1, 1)
        logp = dist.log_prob(action.unsqueeze(0) if action.dim() == 1 else action).sum(-1).squeeze(0)
        return action, logp, value.squeeze(0)

    def push(self, obs, action, logp, reward, value, done) -> None:
        self.buf.append(
            {
                "obs": obs.detach(),
                "action": action.detach(),
                "logp": logp.detach(),
                "reward": float(reward),
                "value": value.detach(),
                "done": bool(done),
            }
        )

    def update(self) -> dict:
        if len(self.buf) < 8:
            self.buf.clear()
            return {"loss": 0.0}
        cfg = self.cfg
        obs = torch.stack([b["obs"] for b in self.buf])
        act = torch.stack([b["action"] for b in self.buf])
        old_logp = torch.stack([b["logp"] for b in self.buf])
        rewards = [b["reward"] for b in self.buf]
        values = torch.stack([b["value"] for b in self.buf])
        dones = [b["done"] for b in self.buf]

        # GAE-lite returns
        rets = []
        R = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + cfg.gamma * R
            rets.append(R)
        rets = torch.tensor(list(reversed(rets)), dtype=torch.float32)
        adv = rets - values.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        last_loss = 0.0
        for _ in range(cfg.epochs):
            dist, v_pred = self.net(obs)
            logp = dist.log_prob(act).sum(-1)
            ratio = (logp - old_logp).exp()
            surr1 = ratio * adv
            surr2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(v_pred, rets)
            ent = dist.entropy().sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - cfg.ent_coef * ent
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            last_loss = float(loss)
        self.buf.clear()
        return {"loss": last_loss}

    def state_dict(self) -> dict:
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict()}

    def load_state_dict(self, payload: dict) -> None:
        self.net.load_state_dict(payload["net"])
        if "opt" in payload:
            self.opt.load_state_dict(payload["opt"])


def self_check() -> None:
    n = 4
    dim = EvaderPPO.obs_dim_for(n)
    ppo = EvaderPPO(dim)
    obs = torch.randn(dim)
    a, lp, v = ppo.act(obs)
    assert a.shape == (2,)
    for _ in range(16):
        ppo.push(obs, a, lp, 0.1, v, False)
    stats = ppo.update()
    print(f"evader_ppo: OK loss={stats['loss']:.4f}")


if __name__ == "__main__":
    self_check()
