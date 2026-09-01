"""Online adaptation with clip_grad_norm_ and param-norm cap.

Entry:
  python -m secdo.training.online_adapt --config configs/secdo/online_adapt.yaml
"""


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.nn.utils import clip_grad_norm_

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig, UAVBandwidthEnv
from secdo.models.dynamics.predictor import SecdoPredictor
from secdo.training.losses import constraint_loss
from secdo.optimizer.secdo_optimizer import SecdoOptimizer
from utils.checkpoint import load_checkpoint, save_checkpoint
from secdo.utils.device import build_device_context, print_training_banner, to_device
from utils.seed import set_seed


@dataclass
class OnlineAdaptConfig:
    lr: float = 1e-3
    zeta: float = 0.05
    device: str = "cuda:0"
    amp: bool = False
    horizon: int = 96
    batch: int = 8
    steps: int = 200
    project_params: bool = True


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@torch.no_grad()
def _project_delta(module: torch.nn.Module, before: dict[str, torch.Tensor], zeta: float) -> None:
    sq = 0.0
    diffs = {}
    for name, p in module.named_parameters():
        d = p - before[name]
        diffs[name] = d
        sq += float(d.pow(2).sum())
    norm = sq**0.5
    if norm <= zeta or norm < 1e-12:
        return
    scale = zeta / norm
    for name, p in module.named_parameters():
        p.copy_(before[name] + diffs[name] * scale)


def adapt_step(
    head: torch.nn.Module,
    c_hat: torch.Tensor,
    c_true: torch.Tensor,
    cfg: OnlineAdaptConfig,
) -> float:
    if not c_hat.requires_grad:
        raise RuntimeError("adapt_step expects c_hat with grad")
    before = {n: p.detach().clone() for n, p in head.named_parameters()}
    loss = constraint_loss(c_hat, c_true)
    head.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(head.parameters(), max_norm=cfg.zeta)
    with torch.no_grad():
        for p in head.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-cfg.lr)
        if cfg.project_params:
            _project_delta(head, before, cfg.zeta)
    return float(loss.detach())


def adapt_step_residual(
    head: torch.nn.Module,
    h: torch.Tensor,
    c_curr: torch.Tensor,
    c_true: torch.Tensor,
    cfg: OnlineAdaptConfig,
) -> float:
    return adapt_step(head, head(h, c_curr), c_true, cfg)


def run_online_loop(
    env_cfg: UAVBandwidthConfig,
    model: SecdoPredictor,
    cfg: OnlineAdaptConfig,
    eta: float = 0.25,
) -> list[dict]:
    ctx = build_device_context(prefer="cuda:0", amp=cfg.amp)
    print_training_banner(ctx, projection="anticipatory", projection_detach=True)
    model = model.to(ctx.device)
    env = UAVBandwidthEnv(env_cfg, device=ctx.device)
    opt_mod = SecdoOptimizer(eta=eta)
    logs: list[dict] = []

    obs = env.reset(batch=cfg.batch)
    n = env_cfg.n_agents
    pref = torch.ones(cfg.batch, n, device=ctx.device) / n
    from secdo.optimizer.anticipatory_projection import reactive_project

    x = reactive_project(pref.clone(), obs["c_teacher"])
    h = None
    t = 0
    while t < cfg.steps and not env.done:
        feat = to_device(obs["feat"], ctx.device)
        c_t = to_device(obs["c_teacher"], ctx.device)
        out = model.forward_step(feat, c_t, h)
        h = out["h"].detach()
        x = opt_mod.step(x, pref, out["c_hat"], detach_budget=True)
        obs = env.step()
        c_next = to_device(obs["c_teacher"], ctx.device)
        c_hat_g = model.constraint_head(out["h"].detach(), c_t)
        lc = adapt_step(model.constraint_head, c_hat_g, c_next, cfg)
        with torch.no_grad():
            delta = float((out["c_hat"] - c_next).abs().mean())
            rho = float((c_next - c_t).abs().mean())
            viol = float((x.sum(-1) - c_next.view(-1)).clamp(min=0).mean())
        logs.append({"t": t, "L_c": lc, "delta": delta, "rho": rho, "viol": viol})
        t += 1
    if logs:
        print_training_banner(
            ctx,
            projection="anticipatory",
            projection_detach=True,
            epsilon=None,
            delta=logs[-1]["delta"],
            violation=logs[-1]["viol"],
        )
    return logs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SECDO online adaptation")
    p.add_argument("--config", type=str, default="configs/secdo/online_adapt.yaml")
    args = p.parse_args(argv)
    raw = _load_yaml(args.config)
    oa = raw.get("online_adapt", {})
    env = raw.get("env", {})
    cfg = OnlineAdaptConfig(
        lr=float(oa.get("lr", 1e-3)),
        zeta=float(oa.get("zeta", 0.05)),
        device="cuda:0",
        batch=int(env.get("batch", 8)),
        steps=int(oa.get("steps", 200)),
        project_params=bool(oa.get("project_params", True)),
    )
    set_seed(int(raw.get("seed", 0)))
    env_cfg = UAVBandwidthConfig(
        regime=str(env.get("regime", "fast")),
        horizon=int(env.get("horizon", 96)),
        n_agents=int(env.get("n_agents", 8)),
    )
    tmp = UAVBandwidthEnv(env_cfg, device="cpu")
    obs = tmp.reset(batch=1)
    model = SecdoPredictor(int(obs["feat"].shape[-1]), residual=True)
    ckpt = raw.get("load_predictor", "")
    if ckpt and Path(ckpt).is_file():
        payload = load_checkpoint(ckpt, map_location="cpu")
        model.load_state_dict(payload["model"])
        print("Loaded", ckpt, flush=True)
    logs = run_online_loop(env_cfg, model, cfg, eta=float(raw.get("optim", {}).get("eta", 0.25)))
    if logs:
        print("online last:", logs[-1], flush=True)
    save_checkpoint(
        Path(raw.get("ckpt_dir", "checkpoints/secdo_online")) / "last.pt",
        {"model": model.state_dict(), "logs": logs},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
