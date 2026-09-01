"""SECDO Phase 2 closed-loop runner (cleaner eval)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig, UAVBandwidthEnv
from experiments.secdo_uav.metrics import (
    TheoryLogger,
    allocation_objective,
    instantaneous_optimum,
)
from secdo.api import (
    ConstraintHead,
    FeasibleHead,
    LatentEncoder,
    anticipatory_project,
    reactive_project,
    step_positions,
    system_capacity_teacher,
)

Method = Literal["secdo", "reactive", "oracle", "no_latent", "no_constraint_dyn"]


@dataclass
class RunConfig:
    method: Method = "secdo"
    eta: float = 0.25
    train_steps: int = 400
    batch: int = 24
    lr: float = 1e-3
    lambda_c: float = 2.0
    lambda_s: float = 0.15
    seed: int = 0
    device: str = "cpu"
    include_series: bool = False


class TinyRecurrent(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros_like(z)
        return self.gru(z, h)


def _peek_c_next(env: UAVBandwidthEnv) -> torch.Tensor:
    assert env.pos is not None and env.vel is not None
    pos2, vel2 = env.pos.clone(), env.vel.clone()
    scale = env._regime_velocity_scale()
    speed = vel2.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vel2 = vel2 / speed * scale
    pos2, _ = step_positions(pos2, vel2, dt=env.cfg.dt, box=env.cfg.box)
    return system_capacity_teacher(pos2, env.channel)


def train_predictors(env: UAVBandwidthEnv, cfg: RunConfig):
    device = torch.device(cfg.device)
    obs = env.reset(batch=cfg.batch)
    feat_dim = int(obs["feat"].shape[-1])
    latent = 32
    enc = LatentEncoder(feat_dim, latent_dim=latent).to(device)
    rec = TinyRecurrent(latent).to(device)
    head = ConstraintHead(feat_dim=latent, c_dim=1, hidden=64, residual=True).to(device)
    dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Linear(64, feat_dim)).to(device)
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(rec.parameters()) + list(head.parameters()) + list(dec.parameters()),
        lr=cfg.lr,
    )
    for _ in range(cfg.train_steps):
        obs = env.reset(batch=cfg.batch)
        h = None
        loss = 0.0
        steps = min(48, env.cfg.horizon - 1)
        for _t in range(steps):
            z = enc(obs["feat"])
            h = rec(z, h)
            c_hat = head(h, obs["c_teacher"])
            s_hat = dec(h)
            obs = env.step()
            loss = loss + cfg.lambda_c * F.mse_loss(c_hat, obs["c_teacher"])
            loss = loss + cfg.lambda_s * F.mse_loss(s_hat, obs["feat"])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc, rec, head, dec, feat_dim, latent


@torch.no_grad()
def evaluate(env: UAVBandwidthEnv, cfg: RunConfig, modules) -> dict:
    enc, rec, head, dec, feat_dim, latent = modules
    device = torch.device(cfg.device)
    n = env.cfg.n_agents
    fhead = FeasibleHead("sum_budget")
    logger = TheoryLogger()

    obs = env.reset(batch=cfg.batch)
    pref = torch.ones(cfg.batch, n, device=device) / n
    x = reactive_project(pref.clone(), obs["c_teacher"])
    h = None
    method = cfg.method

    while not env.done:
        feat = obs["feat"]
        c_t = obs["c_teacher"]

        z = enc(feat)
        if method == "no_latent":
            h_used = z
            s_hat = feat
            eps_t = 1.0
        else:
            h = rec(z, h)
            h_used = h
            s_hat = dec(h_used)
            eps_t = float((feat - s_hat).norm(dim=-1).mean().item())

        if method == "oracle":
            c_hat = _peek_c_next(env)
        elif method in ("reactive", "no_constraint_dyn"):
            c_hat = c_t
        else:
            c_hat = head(h_used, c_t)

        y = x - cfg.eta * (x - pref)
        if method in ("reactive", "no_constraint_dyn"):
            x = reactive_project(y, c_t)
        else:
            x = anticipatory_project(y, fhead(c_hat))

        obs = env.step()
        c_next = obs["c_teacher"]
        rho_val = float((c_next - c_t).abs().mean().item())
        delta_t = float((c_hat - c_next).abs().mean().item())
        if method == "no_latent":
            eps_t = max(eps_t, 0.5)

        viol = float((x.sum(-1) - c_next.view(-1)).clamp(min=0).mean().item())
        x_star = instantaneous_optimum(c_next, n, pref)
        gap = float(
            (allocation_objective(x, pref) - allocation_objective(x_star, pref)).clamp(min=0).mean().item()
        )
        logger.update(
            eps_t=eps_t,
            delta_t=delta_t,
            rho_t=rho_val,
            viol_t=viol,
            gap_t=gap,
            F_t=float(allocation_objective(x, pref).mean().item()),
        )

    out = logger.summary(include_series=cfg.include_series)
    out["method"] = method
    out["regime"] = env.cfg.regime
    return out


def run_once(env_cfg: UAVBandwidthConfig, run_cfg: RunConfig) -> dict:
    env = UAVBandwidthEnv(env_cfg, device=run_cfg.device)
    mods = train_predictors(env, run_cfg)
    env_eval = UAVBandwidthEnv(env_cfg, device=run_cfg.device)
    return evaluate(env_eval, run_cfg, mods)
