"""Stage I - predictor pretraining (cuda:0 / AMP FP16 / DDP).

Entry:
  python -m secdo.training.pretrain_predictor --config configs/secdo/predictor_pretrain.yaml
"""


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc, assignment]

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig
from secdo.models.dynamics.predictor import SecdoPredictor
from secdo.training.losses import constraint_loss, dynamics_loss, predictor_loss
from utils.checkpoint import load_checkpoint, save_best_last, save_checkpoint
from secdo.training.dataset import UAVTrajectoryDataset, collate_pad
from secdo.utils.device import (
    build_device_context,
    maybe_barrier,
    print_training_banner,
    to_device,
    unwrap,
    wrap_ddp,
)
from secdo.utils.metrics import TheoryMeters
from utils.seed import set_seed


@dataclass
class PretrainConfig:
    epochs: int = 50
    steps_per_epoch: int = 40
    batch: int = 24
    lr: float = 1e-4
    lambda_c: float = 1.0
    lambda_s: float = 0.5
    lambda_u: float = 0.1
    roll_len: int = 48
    seed: int = 0
    device: str = "cuda:0"
    amp: bool = True
    latent_dim: int = 32
    n_train_traj: int = 256
    n_val_traj: int = 64
    num_workers: int = 0
    log_dir: str = "runs/secdo_pretrain"
    ckpt_dir: str = "checkpoints/secdo_predictor"
    cosine: bool = True
    use_online_env: bool = False


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_from_yaml(path: str | Path) -> tuple[PretrainConfig, UAVBandwidthConfig]:
    raw = _load_yaml(path)
    loss = raw.get("loss", {})
    opt = raw.get("optimizer", {})
    sch = raw.get("scheduler", {})
    data = raw.get("data", {})
    env = raw.get("env", {})
    cfg = PretrainConfig(
        epochs=int(raw.get("epochs", 50)),
        steps_per_epoch=int(raw.get("steps_per_epoch", 40)),
        batch=int(raw.get("batch", 24)),
        lr=float(opt.get("lr", raw.get("lr", 1e-4))),
        lambda_c=float(loss.get("lambda_c", loss.get("lambda_constraint", 1.0))),
        lambda_s=float(loss.get("lambda_s", loss.get("lambda_state", 0.5))),
        lambda_u=float(loss.get("lambda_u", loss.get("lambda_utility", 0.1))),
        roll_len=int(raw.get("roll_len", 48)),
        seed=int(raw.get("seed", 0)),
        device=str(raw.get("device", "cuda:0")),
        amp=bool(raw.get("amp", True)),
        latent_dim=int(raw.get("latent_dim", 32)),
        n_train_traj=int(data.get("train_trajectories", 256)),
        n_val_traj=int(data.get("val_trajectories", 64)),
        log_dir=str(raw.get("log_dir", "runs/secdo_pretrain")),
        ckpt_dir=str(raw.get("ckpt_dir", "checkpoints/secdo_predictor")),
        cosine=bool(sch.get("cosine", True)),
        use_online_env=bool(raw.get("use_online_env", False)),
    )
    env_cfg = UAVBandwidthConfig(
        regime=str(env.get("regime", "fast")),
        horizon=int(env.get("horizon", 96)),
        n_agents=int(env.get("n_agents", 8)),
        seed=cfg.seed,
    )
    return cfg, env_cfg


def build_predictor(feat_dim: int, cfg: PretrainConfig) -> SecdoPredictor:
    return SecdoPredictor(feat_dim=feat_dim, latent_dim=cfg.latent_dim, residual=True)


def _eps_l2(s_hat: torch.Tensor, s_true: torch.Tensor) -> float:
    """Theory ε proxy: mean ||s - ŝ||_2 (not raw MSE on unscaled features)."""
    return float((s_hat.float() - s_true.float()).norm(dim=-1).mean())


@torch.no_grad()
def _eval_delta_eps(model: SecdoPredictor, loader: DataLoader, ctx) -> tuple[float, float, float]:
    model.eval()
    meters = TheoryMeters()
    for batch in loader:
        batch = to_device(batch, ctx.device)
        feat, c = batch["feat"], batch["c"]
        if c.dim() == 2:
            c = c.unsqueeze(-1)
        T = feat.shape[0]
        h = None
        for t in range(T - 1):
            out = model.forward_step(feat[t], c[t], h)
            h = out["h"]
            meters.update(
                delta=float(constraint_loss(out["c_hat"], c[t + 1])),
                eps=_eps_l2(out["s_hat"], feat[t + 1]),
                loss_c=float(constraint_loss(out["c_hat"], c[t + 1])),
            )
    return meters.mean("eps"), meters.mean("delta"), meters.mean("loss_c")


def pretrain_predictor(
    env_cfg: UAVBandwidthConfig,
    cfg: PretrainConfig,
) -> tuple[dict, list[dict]]:
    set_seed(cfg.seed)
    ctx = build_device_context(prefer="cuda:0", amp=cfg.amp)
    print_training_banner(ctx, projection="n/a (Stage I �?no Π)", projection_detach=True)

    probe = UAVTrajectoryDataset(env_cfg, n_traj=1, seed=cfg.seed, device="cpu")
    feat_dim = int(probe[0]["feat"].shape[-1])
    model = wrap_ddp(build_predictor(feat_dim, cfg), ctx)
    raw = unwrap(model)

    train_ds = UAVTrajectoryDataset(env_cfg, n_traj=cfg.n_train_traj, seed=cfg.seed, device="cpu")
    val_ds = UAVTrajectoryDataset(
        env_cfg, n_traj=max(1, cfg.n_val_traj), seed=cfg.seed + 10_000, device="cpu"
    )
    sampler = DistributedSampler(train_ds, shuffle=True) if ctx.use_ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, cfg.batch // max(ctx.world_size, 1)),
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_pad,
        num_workers=cfg.num_workers,
        pin_memory=(ctx.device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, cfg.batch // max(ctx.world_size, 1)),
        shuffle=False,
        collate_fn=collate_pad,
        num_workers=0,
        pin_memory=(ctx.device.type == "cuda"),
    )

    opt = torch.optim.AdamW(raw.parameters(), lr=cfg.lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
        if cfg.cosine
        else None
    )
    scaler = GradScaler("cuda", enabled=ctx.amp)
    writer = SummaryWriter(cfg.log_dir) if (ctx.is_main and SummaryWriter is not None) else None

    logs: list[dict] = []
    best_delta = float("inf")
    for ep in range(cfg.epochs):
        if sampler is not None:
            sampler.set_epoch(ep)
        meters = TheoryMeters()
        model.train()
        for batch in train_loader:
            batch = to_device(batch, ctx.device)
            feat, c = batch["feat"], batch["c"]
            if c.dim() == 2:
                c = c.unsqueeze(-1)
            T = feat.shape[0]
            h = None
            loss = torch.zeros((), device=ctx.device)
            with autocast("cuda", enabled=ctx.amp):
                for t in range(T - 1):
                    out = raw.forward_step(feat[t], c[t], h)
                    h = out["h"]
                    loss = loss + predictor_loss(
                        out["s_hat"],
                        feat[t + 1],
                        out["c_hat"],
                        c[t + 1],
                        lambda_s=cfg.lambda_s,
                        lambda_c=cfg.lambda_c,
                        lambda_u=cfg.lambda_u,
                    )
                    with torch.no_grad():
                        meters.update(
                            loss_c=float(constraint_loss(out["c_hat"].float(), c[t + 1])),
                            loss_s=float(dynamics_loss(out["s_hat"].float(), feat[t + 1])),
                            eps=_eps_l2(out["s_hat"], feat[t + 1]),
                            delta=float(constraint_loss(out["c_hat"].float(), c[t + 1])),
                        )
                loss = loss / max(T - 1, 1)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            meters.update(loss_total=float(loss.detach()))

        if sched is not None:
            sched.step()

        eps_v, delta_v, mse_c = _eval_delta_eps(raw, val_loader, ctx)
        log = {
            "epoch": ep,
            "stage": "pretrain",
            "epsilon": eps_v,
            "delta": delta_v,
            "constraint_MSE": mse_c,
            "MSE_c": mse_c,
            "eps_proxy": eps_v,
            "loss_total": meters.mean("loss_total"),
        }
        logs.append(log)
        is_best = delta_v < best_delta
        if is_best:
            best_delta = delta_v

        if ctx.is_main:
            print(
                f"[pretrain] ep={ep} epsilon={eps_v:.6f} delta={delta_v:.6f} "
                f"constraint_MSE={mse_c:.6f} best_delta={best_delta:.6f}",
                flush=True,
            )
            save_best_last(
                cfg.ckpt_dir,
                {
                    "epoch": ep,
                    "model": raw.state_dict(),
                    "feat_dim": feat_dim,
                    "latent_dim": cfg.latent_dim,
                    "metrics": log,
                    "cfg": cfg.__dict__,
                },
                is_best=is_best,
            )
            if writer is not None:
                writer.add_scalar("metric/epsilon", eps_v, ep)
                writer.add_scalar("metric/delta", delta_v, ep)
                writer.add_scalar("train/loss_constraint", mse_c, ep)
                writer.add_scalar("train/loss_total", log["loss_total"], ep)
        maybe_barrier()

    if writer is not None:
        writer.close()
    if ctx.is_main and logs:
        print_training_banner(
            ctx,
            projection="n/a (Stage I)",
            projection_detach=True,
            epsilon=logs[-1]["epsilon"],
            delta=logs[-1]["delta"],
        )
    return raw.as_modules_dict(), logs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SECDO Stage I predictor pretrain")
    p.add_argument("--config", type=str, default="configs/secdo/predictor_pretrain.yaml")
    args = p.parse_args(argv)
    cfg, env_cfg = config_from_yaml(args.config)
    pretrain_predictor(env_cfg, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
