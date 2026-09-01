"""Stage II - closed-loop SECDO with stop-gradient projection (CUDA/AMP/DDP).

Entry:
  python -m secdo.training.train_secdo --config configs/secdo/secdo_joint.yaml
  torchrun --nproc_per_node=4 -m secdo.training.train_secdo --config ...
"""


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch.amp import GradScaler, autocast
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig, UAVBandwidthEnv
from experiments.secdo_uav.metrics import allocation_objective, instantaneous_optimum
from secdo.models.dynamics.predictor import SecdoPredictor
from secdo.training.losses import constraint_loss, dynamics_loss, optimization_loss, predictor_loss, violation
from secdo.optimizer.secdo_optimizer import SecdoOptimizer
from utils.checkpoint import load_checkpoint, save_best_last, save_checkpoint
from secdo.utils.device import (
    build_device_context,
    maybe_barrier,
    print_training_banner,
    unwrap,
    wrap_ddp,
)
from secdo.training.pretrain_predictor import PretrainConfig, pretrain_predictor
from secdo.utils.metrics import TheoryMeters
from utils.seed import set_seed

Mode = Literal["frozen_predictor", "joint", "head_only"]


@dataclass
class SecdoTrainConfig:
    pretrain_epochs: int = 50
    head_only_epochs: int = 50
    joint_epochs: int = 50
    steps_per_epoch: int = 20
    batch: int = 16
    lr: float = 1e-3
    lr_joint: float = 3e-4
    eta: float = 0.25
    gamma_v: float = 1.0
    lambda_pred: float = 1.0
    lambda_c: float = 1.0
    lambda_s: float = 0.5
    lambda_u: float = 0.1
    seed: int = 0
    device: str = "cuda:0"
    amp: bool = True
    skip_pretrain: bool = False
    load_predictor: str = "checkpoints/secdo_predictor/best.pt"
    log_dir: str = "runs/secdo_joint"
    ckpt_dir: str = "checkpoints/secdo_joint"
    stop_gradient_projection: bool = True
    latent_dim: int = 32


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_from_yaml(path: str | Path) -> tuple[SecdoTrainConfig, UAVBandwidthConfig]:
    raw = _load_yaml(path)
    sch = raw.get("schedule", {})
    optim = raw.get("optim", {})
    loss = raw.get("loss", {})
    env = raw.get("env", {})
    cfg = SecdoTrainConfig(
        pretrain_epochs=int(sch.get("pretrain_epochs", 50)),
        head_only_epochs=int(sch.get("head_only_epochs", 50)),
        joint_epochs=int(sch.get("joint_epochs", 50)),
        steps_per_epoch=int(sch.get("steps_per_epoch", 20)),
        batch=int(env.get("batch", 16)),
        lr=float(optim.get("lr", 1e-3)),
        lr_joint=float(optim.get("lr_joint", 3e-4)),
        eta=float(optim.get("eta", 0.25)),
        gamma_v=float(optim.get("gamma_v", 1.0)),
        lambda_pred=float(optim.get("lambda_pred", 1.0)),
        lambda_c=float(loss.get("lambda_c", 1.0)),
        lambda_s=float(loss.get("lambda_s", 0.5)),
        lambda_u=float(loss.get("lambda_u", 0.1)),
        seed=int(raw.get("seed", 0)),
        device=str(raw.get("device", "cuda:0")),
        amp=bool(raw.get("amp", True)),
        skip_pretrain=bool(raw.get("skip_pretrain", False)),
        load_predictor=str(raw.get("load_predictor", "checkpoints/secdo_predictor/best.pt")),
        log_dir=str(raw.get("log_dir", "runs/secdo_joint")),
        ckpt_dir=str(raw.get("ckpt_dir", "checkpoints/secdo_joint")),
        stop_gradient_projection=bool(raw.get("stop_gradient_projection", True)),
        latent_dim=int(raw.get("latent_dim", 32)),
    )
    env_cfg = UAVBandwidthConfig(
        regime=str(env.get("regime", "fast")),
        horizon=int(env.get("horizon", 96)),
        n_agents=int(env.get("n_agents", 8)),
        seed=cfg.seed,
    )
    return cfg, env_cfg


def _set_trainable(model: SecdoPredictor, mode: Mode) -> None:
    if mode == "frozen_predictor":
        for p in model.parameters():
            p.requires_grad = False
    elif mode == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.constraint_head.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True


def _closed_loop_epoch(
    env: UAVBandwidthEnv,
    model: SecdoPredictor,
    cfg: SecdoTrainConfig,
    mode: Mode,
    ctx,
    scaler: GradScaler,
) -> dict:
    assert cfg.stop_gradient_projection, "SECDO requires ĉ.detach() through projection"
    opt_mod = SecdoOptimizer(eta=cfg.eta)
    n = env.cfg.n_agents
    params = [p for p in model.parameters() if p.requires_grad]
    optim = (
        torch.optim.AdamW(params, lr=cfg.lr_joint if mode == "joint" else cfg.lr)
        if params
        else None
    )

    meters = TheoryMeters()
    n_steps = 0
    frac_ok = 0
    sum_lopt = 0.0
    n_roll = 0

    for _ in range(cfg.steps_per_epoch):
        obs = env.reset(batch=cfg.batch)
        pref = torch.ones(cfg.batch, n, device=ctx.device) / n
        from secdo.optimizer.anticipatory_projection import reactive_project

        x = reactive_project(pref.clone(), obs["c_teacher"])
        h = None
        F_list: list[torch.Tensor] = []
        V_list: list[torch.Tensor] = []
        L_pred_acc = torch.zeros((), device=ctx.device)

        while not env.done:
            feat = obs["feat"]
            c_t = obs["c_teacher"]
            with autocast("cuda", enabled=ctx.amp):
                out = model.forward_step(feat, c_t, h)
                h = out["h"]
                c_hat = out["c_hat"]
                s_hat = out["s_hat"]
                # Algorithm 1: NEVER backprop through Π
                x = opt_mod.step(x, pref, c_hat, detach_budget=True)

            obs = env.step()
            c_next = obs["c_teacher"]
            with autocast("cuda", enabled=ctx.amp):
                L_pred_acc = L_pred_acc + predictor_loss(
                    s_hat,
                    obs["feat"],
                    c_hat,
                    c_next,
                    lambda_s=cfg.lambda_s,
                    lambda_c=cfg.lambda_c,
                    lambda_u=cfg.lambda_u,
                )
            F_t = allocation_objective(x, pref).mean()
            V_t = violation(x, c_next)
            F_list.append(F_t.detach())
            V_list.append(V_t.detach())

            with torch.no_grad():
                delta = float((c_hat.float() - c_next).abs().mean())
                rho = float((c_next - c_t).abs().mean())
                eps = float((obs["feat"] - s_hat.float()).norm(dim=-1).mean())
                x_star = instantaneous_optimum(c_next, n, pref)
                gap = float(
                    (allocation_objective(x, pref) - allocation_objective(x_star, pref))
                    .clamp(min=0)
                    .mean()
                )
                meters.update(
                    eps=eps,
                    delta=delta,
                    rho=rho,
                    gap=gap,
                    viol=float(V_t),
                    loss_c=float(constraint_loss(c_hat.float(), c_next)),
                    loss_s=float(dynamics_loss(s_hat.float(), obs["feat"])),
                )
                frac_ok += int(delta < rho)
                n_steps += 1

        L_opt_val = float(
            optimization_loss(torch.stack(F_list), torch.stack(V_list), gamma=cfg.gamma_v)
        )
        sum_lopt += L_opt_val
        n_roll += 1
        meters.update(loss_total=float(L_pred_acc.detach()) + cfg.lambda_pred * L_opt_val)

        if optim is not None and mode != "frozen_predictor":
            optim.zero_grad(set_to_none=True)
            scaler.scale(L_pred_acc).backward()
            scaler.step(optim)
            scaler.update()

    inv = max(n_steps, 1)
    return {
        "mode": mode,
        "Gap": meters.mean("gap"),
        "sum_V": sum(meters.viol),
        "mean_V": meters.mean("viol"),
        "L_opt": sum_lopt / max(n_roll, 1),
        "delta": meters.mean("delta"),
        "rho": meters.mean("rho"),
        "eps": meters.mean("eps"),
        "MSE_c": meters.mean("loss_c"),
        "frac_delta_lt_rho": frac_ok / inv,
        "meters": meters,
    }


def train_secdo(
    env_cfg: UAVBandwidthConfig,
    cfg: SecdoTrainConfig,
    modules: dict | None = None,
) -> tuple[dict, list[dict]]:
    set_seed(cfg.seed)
    ctx = build_device_context(prefer="cuda:0", amp=cfg.amp)
    assert cfg.stop_gradient_projection, "must use c_hat.detach() before projection"

    logs: list[dict] = []
    writer = SummaryWriter(cfg.log_dir) if (ctx.is_main and SummaryWriter is not None) else None

    env = UAVBandwidthEnv(env_cfg, device=ctx.device)
    obs = env.reset(batch=1)
    feat_dim = int(obs["feat"].shape[-1])

    if modules is None and not cfg.skip_pretrain:
        # Prefer frozen Stage-I checkpoint; else run short embedded pretrain
        ckpt_path = Path(cfg.load_predictor)
        if ckpt_path.is_file():
            from utils.checkpoint import load_checkpoint

            model = SecdoPredictor(feat_dim, latent_dim=cfg.latent_dim, residual=True)
            payload = load_checkpoint(ckpt_path, map_location="cpu")
            model.load_state_dict(payload["model"])
            if ctx.is_main:
                print(f"Loaded Stage-I predictor: {ckpt_path}", flush=True)
        else:
            pcfg = PretrainConfig(
                epochs=cfg.pretrain_epochs,
                batch=cfg.batch,
                lr=cfg.lr,
                lambda_c=cfg.lambda_c,
                lambda_s=cfg.lambda_s,
                lambda_u=cfg.lambda_u,
                seed=cfg.seed,
                device="cuda:0",
                amp=cfg.amp,
                latent_dim=cfg.latent_dim,
                n_train_traj=max(64, cfg.batch * 4),
                n_val_traj=32,
                ckpt_dir="checkpoints/secdo_predictor",
            )
            modules, plogs = pretrain_predictor(env_cfg, pcfg)
            logs.extend(plogs)
            model = modules["predictor"]
    elif modules is not None and "predictor" in modules:
        model = modules["predictor"]
    else:
        model = SecdoPredictor(feat_dim, latent_dim=cfg.latent_dim, residual=True)
        if modules is not None:
            model.encoder.load_state_dict(modules["enc"].state_dict())
            model.gru.load_state_dict(modules["rec"].state_dict())
            model.constraint_head.load_state_dict(modules["head"].state_dict())
            model.state_decoder.load_state_dict(modules["dec"].state_dict())

    model = wrap_ddp(model, ctx)
    raw = unwrap(model)
    scaler = GradScaler("cuda", enabled=ctx.amp)
    global_ep = 0
    best_gap = float("inf")

    schedule = [
        ("frozen_predictor", max(1, min(3, cfg.head_only_epochs))),
        ("head_only", cfg.head_only_epochs),
        ("joint", cfg.joint_epochs),
    ]
    print_training_banner(ctx, projection="anticipatory", projection_detach=True)

    for stage, n_ep in schedule:
        _set_trainable(raw, stage)  # type: ignore[arg-type]
        for ep in range(n_ep):
            m = _closed_loop_epoch(env, raw, cfg, stage, ctx, scaler)  # type: ignore[arg-type]
            row = {k: v for k, v in m.items() if k != "meters"}
            row.update({"epoch": ep, "stage": stage})
            logs.append(row)
            if writer is not None:
                for k, v in m["meters"].as_tb_dict().items():
                    writer.add_scalar(f"{stage}/{k}", v, global_ep)
                writer.add_scalar(f"{stage}/frac_delta_lt_rho", row["frac_delta_lt_rho"], global_ep)
            if ctx.is_main:
                print(
                    f"[{stage}] ep={ep} epsilon={row['eps']:.6f} delta={row['delta']:.6f} "
                    f"violation={row['mean_V']:.6f} gap={row['Gap']:.6f} "
                    f"frac(δ<ϝ)={row['frac_delta_lt_rho']:.2f}",
                    flush=True,
                )
                is_best = row["Gap"] < best_gap
                if is_best:
                    best_gap = row["Gap"]
                save_best_last(
                    cfg.ckpt_dir,
                    {"stage": stage, "epoch": ep, "model": raw.state_dict(), "metrics": row},
                    is_best=is_best,
                )
                save_checkpoint(
                    Path(cfg.ckpt_dir) / f"{stage}_epoch_{ep:04d}.pt",
                    {"stage": stage, "epoch": ep, "model": raw.state_dict()},
                )
            global_ep += 1
            maybe_barrier()

    if writer is not None:
        writer.close()
    if logs and ctx.is_main:
        last = logs[-1]
        print_training_banner(
            ctx,
            projection="anticipatory",
            projection_detach=True,
            epsilon=last["eps"],
            delta=last["delta"],
            violation=last["mean_V"],
        )
    return raw.as_modules_dict(), logs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SECDO Stage II joint training")
    p.add_argument("--config", type=str, default="configs/secdo/secdo_joint.yaml")
    args = p.parse_args(argv)
    cfg, env_cfg = config_from_yaml(args.config)
    train_secdo(env_cfg, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
