"""Alternating high/low REINFORCE trainer for hierarchical role control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from dice.role_reward import RoleReward

from .utils import freeze, light_scout_ratio, pack_actions, save_checkpoint, unfreeze


class HierarchicalTrainer:
    def __init__(self, env: Any, high_policy: Any, low_policy: Any, config: dict):
        self.env = env
        self.high_policy = high_policy
        self.low_policy = low_policy
        self.config = config
        self.high_optim = torch.optim.Adam(high_policy.parameters(), lr=config.get("high_lr", 3e-4))
        self.low_optim = torch.optim.Adam(low_policy.parameters(), lr=config.get("low_lr", 3e-4))
        self.gamma = float(config.get("gamma", 0.95))
        self.ep_max = int(config.get("max_steps", 200))
        self.metrics: dict[str, list] = {
            "step": [],
            "coverage": [],
            "collision": [],
            "role_entropy": [],
            "light_scout": [],
        }
        self.save_dir = Path(config.get("save_dir", "experiment_results/hierarchical/"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._global_step = 0
        self._role_reward = RoleReward(int(config.get("n_roles", 3)))
        self.match_weight = float(config.get("match_weight", 1.0))
        self.coll_coef = float(config.get("coll_coef", 1.0))

    def _discounted_returns(self, rewards: list[torch.Tensor]) -> list[torch.Tensor]:
        n = rewards[0].shape[0]
        G = torch.zeros(n)
        out: list[torch.Tensor] = []
        for rw in reversed(rewards):
            G = rw + self.gamma * G
            out.append(G)
        return list(reversed(out))

    def _log_metrics(self, info: dict, roles: torch.Tensor) -> None:
        self.metrics["step"].append(self._global_step)
        self.metrics["coverage"].append(float(info.get("coverage", 0.0)))
        self.metrics["collision"].append(float(info.get("collision_rate", 0.0)))
        logits = self.high_policy.mean_logits
        probs = torch.softmax(logits, dim=-1)
        ent = float(-(probs * torch.log(probs + 1e-8)).sum())
        self.metrics["role_entropy"].append(ent)
        self.metrics["light_scout"].append(light_scout_ratio(roles, self.env.uav_type_list))

    def train_high(self, env_steps: int) -> None:
        """Freeze low, train high with episode REINFORCE on global reward."""
        freeze(self.low_policy)
        unfreeze(self.high_policy)
        start = self._global_step
        ep = 0
        while self._global_step - start < env_steps:
            obs, _ = self.env.reset(seed=self.config.get("seed", 0) + ep)
            roles: torch.Tensor | None = None
            high_lps: list[torch.Tensor] = []
            info: dict = {}
            done = False
            while not done and self._global_step - start < env_steps:
                need = roles is None or (self.env.step_count % self.high_policy.decision_interval == 0)
                if need:
                    gobs = self.env.get_global_obs()
                    self.high_policy.update_logits(
                        gobs,
                        self.env.uav_type_list,
                        step=self.env.step_count,
                        force=roles is None,
                    )
                    roles, lp = self.high_policy.get_roles(self.env.n_agents)
                    high_lps.append(lp)
                assert roles is not None
                with torch.no_grad():
                    dxdy = self.low_policy(obs, roles)
                actions = pack_actions(dxdy, roles, self.env.n_roles)
                obs, _r, done, trunc, info = self.env.step(actions)
                self._global_step += 1
                if trunc or done:
                    done = True

            if high_lps and roles is not None:
                cov = float(info.get("coverage", 0.0))
                coll = float(info.get("collision_rate", 0.0))
                lsr = light_scout_ratio(roles, self.env.uav_type_list)
                match = float(
                    self._role_reward.capability_match_bonus(roles, self.env.uav_type_list).mean()
                )
                R = cov - self.coll_coef * coll + 0.5 * lsr + self.match_weight * match
                bonus = float(self.config.get("role_entropy_bonus", 0.0))
                assert self.high_policy._logits is not None
                probs = torch.softmax(self.high_policy._logits.mean(dim=0), dim=-1)
                ent = -(probs * torch.log(probs + 1e-8)).sum()
                loss = -sum(high_lps) * R - bonus * ent
                self.high_optim.zero_grad()
                loss.backward()
                self.high_optim.step()

            if roles is not None:
                self._log_metrics(info, roles)
            if ep % 25 == 0 or self._global_step - start >= env_steps:
                print(
                    f"[high] ep={ep} step={self._global_step} "
                    f"cov={info.get('coverage', 0):.2f} "
                    f"coll={info.get('collision_rate', 0):.3f} "
                    f"ent={self.metrics['role_entropy'][-1]:.3f} "
                    f"light_scout={self.metrics['light_scout'][-1]:.2f}",
                    flush=True,
                )
            ep += 1

    def train_low(self, env_steps: int) -> None:
        """Freeze high, train low with per-agent REINFORCE on env rewards."""
        freeze(self.high_policy)
        unfreeze(self.low_policy)
        start = self._global_step
        ep = 0
        while self._global_step - start < env_steps:
            obs, _ = self.env.reset(seed=self.config.get("seed", 0) + 10_000 + ep)
            with torch.no_grad():
                gobs = self.env.get_global_obs()
                self.high_policy.update_logits(gobs, self.env.uav_type_list, step=0, force=True)
                fixed_roles, _ = self.high_policy.get_roles(self.env.n_agents)
            logps: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            info: dict = {}
            done = False
            while not done and self._global_step - start < env_steps:
                dxdy, lp = self.low_policy.act(obs, fixed_roles)
                actions = pack_actions(dxdy, fixed_roles, self.env.n_roles)
                obs, r, done, trunc, info = self.env.step(actions.detach())
                logps.append(lp)
                rewards.append(r.detach().cpu())
                self._global_step += 1
                if trunc or done:
                    done = True

            if logps:
                returns = self._discounted_returns(rewards)
                loss = torch.zeros(())
                for lp, ret in zip(logps, returns):
                    loss = loss - (lp * ret).mean()
                self.low_optim.zero_grad()
                loss.backward()
                self.low_optim.step()

            self._log_metrics(info, fixed_roles)
            if ep % 25 == 0 or self._global_step - start >= env_steps:
                print(
                    f"[low] ep={ep} step={self._global_step} "
                    f"cov={info.get('coverage', 0):.2f} "
                    f"coll={info.get('collision_rate', 0):.3f} "
                    f"ent={self.metrics['role_entropy'][-1]:.3f} "
                    f"light_scout={self.metrics['light_scout'][-1]:.2f}",
                    flush=True,
                )
            ep += 1

    def alternate_train(self, rounds: int, high_steps: int, low_steps: int) -> None:
        for r in range(rounds):
            print(f"=== round {r}: high ({high_steps} env steps) ===", flush=True)
            self.train_high(high_steps)
            save_checkpoint(self.high_policy, self.save_dir / f"round_{r}_high.pt")
            print(f"=== round {r}: low ({low_steps} env steps) ===", flush=True)
            self.train_low(low_steps)
            save_checkpoint(self.low_policy, self.save_dir / f"round_{r}_low.pt")
        self.save_metrics()

    def save_metrics(self) -> None:
        path = self.save_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"metrics → {path}", flush=True)
