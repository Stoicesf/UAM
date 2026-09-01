"""Unified SECDO Trainer — modes: pretrain | joint | online.

Handles AMP, checkpoint best/last, logging, resume.
Projection always uses ĉ.detach() in joint/online.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from experiments.secdo_uav.env_uav_bandwidth import UAVBandwidthConfig, UAVBandwidthEnv
from secdo.datasets.loaders import TrajectoryDataset, collate_trajectories
from secdo.datasets.uav.generator import GenerateConfig, generate_dataset
from secdo.models import SECDO
from secdo.optimizer import reactive_project
from secdo.training.losses import (
    allocation_objective,
    constraint_loss,
    dynamics_loss,
    eps_l2,
    predictor_loss,
    violation,
)
from secdo.utils.checkpoint import load_checkpoint, save_best_last
from secdo.utils.device import build_device_context, print_banner, to_device, unwrap, wrap_ddp
from secdo.utils.logging import TheoryLog, log_scalars, make_writer
from secdo.utils.seed import set_seed

TrainMode = Literal["pretrain", "joint", "online"]


@dataclass
class TrainConfig:
    mode: TrainMode = "pretrain"
    seed: int = 0
    amp: bool = True
    # model
    latent_dim: int = 32
    hidden: int = 64
    eta: float = 0.25
    n_agents: int = 8
    # loss
    lambda_s: float = 0.5
    lambda_c: float = 1.0
    lambda_u: float = 0.1
    lambda_pred: float = 1.0
    gamma_v: float = 1.0
    # optim
    lr: float = 1e-4
    lr_joint: float = 3e-4
    cosine: bool = True
    # schedule
    epochs: int = 8
    batch: int = 16
    steps_per_epoch: int = 8
    head_only_epochs: int = 3
    joint_epochs: int = 3
    frozen_epochs: int = 2
    # data
    data_dir: str = ""
    n_train_traj: int = 128
    n_val_traj: int = 32
    traj_length: int = 48
    regime: str = "fast"
    horizon: int = 48
    # io
    ckpt_dir: str = "checkpoints/secdo_platform"
    log_dir: str = "runs/secdo_platform"
    load_predictor: str = ""
    resume: str = ""
    # online
    zeta: float = 0.05
    online_steps: int = 100
    extras: dict[str, Any] = field(default_factory=dict)


def load_train_config(path: str | Path, mode: TrainMode | None = None) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    loss = raw.get("loss", {})
    optim = raw.get("optim", raw.get("optimizer", {}))
    sch = raw.get("schedule", {})
    env = raw.get("env", {})
    data = raw.get("data", {})
    oa = raw.get("online_adapt", {})
    cfg = TrainConfig(
        mode=mode or raw.get("mode", "pretrain"),  # type: ignore[arg-type]
        seed=int(raw.get("seed", 0)),
        amp=bool(raw.get("amp", True)),
        latent_dim=int(raw.get("latent_dim", 32)),
        hidden=int(raw.get("hidden", 64)),
        eta=float(optim.get("eta", raw.get("eta", 0.25))),
        n_agents=int(env.get("n_agents", 8)),
        lambda_s=float(loss.get("lambda_s", loss.get("lambda_state", 0.5))),
        lambda_c=float(loss.get("lambda_c", loss.get("lambda_constraint", 1.0))),
        lambda_u=float(loss.get("lambda_u", loss.get("lambda_utility", 0.1))),
        lambda_pred=float(optim.get("lambda_pred", 1.0)),
        gamma_v=float(optim.get("gamma_v", 1.0)),
        lr=float(optim.get("lr", 1e-4)),
        lr_joint=float(optim.get("lr_joint", 3e-4)),
        cosine=bool(raw.get("scheduler", {}).get("cosine", True)),
        epochs=int(raw.get("epochs", sch.get("pretrain_epochs", 8))),
        batch=int(raw.get("batch", env.get("batch", 16))),
        steps_per_epoch=int(sch.get("steps_per_epoch", raw.get("steps_per_epoch", 8))),
        head_only_epochs=int(sch.get("head_only_epochs", 3)),
        joint_epochs=int(sch.get("joint_epochs", 3)),
        frozen_epochs=int(sch.get("frozen_epochs", 2)),
        data_dir=str(data.get("dir", raw.get("data_dir", ""))),
        n_train_traj=int(data.get("train_trajectories", 128)),
        n_val_traj=int(data.get("val_trajectories", 32)),
        traj_length=int(data.get("length", env.get("horizon", 48))),
        regime=str(env.get("regime", "fast")),
        horizon=int(env.get("horizon", 48)),
        ckpt_dir=str(raw.get("ckpt_dir", "checkpoints/secdo_platform")),
        log_dir=str(raw.get("log_dir", "runs/secdo_platform")),
        load_predictor=str(raw.get("load_predictor", "")),
        resume=str(raw.get("resume", "")),
        zeta=float(oa.get("zeta", 0.05)),
        online_steps=int(oa.get("steps", 100)),
    )
    return cfg


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.ctx = build_device_context(amp=cfg.amp)
        self.writer = make_writer(cfg.log_dir) if self.ctx.is_main else None
        self.scaler = GradScaler("cuda", enabled=self.ctx.amp)
        self.model: SECDO | None = None
        self.logs: list[dict] = []

    def _ensure_offline_data(self) -> Path:
        if self.cfg.data_dir and Path(self.cfg.data_dir).is_dir():
            return Path(self.cfg.data_dir)
        out = Path("data/secdo/uav_train_cache")
        if (out / "trajectories.pt").is_file():
            return out
        if self.ctx.is_main:
            print(f"Generating offline cache → {out}", flush=True)
            generate_dataset(
                out,
                GenerateConfig(
                    trajectories=self.cfg.n_train_traj + self.cfg.n_val_traj,
                    length=self.cfg.traj_length,
                    seed=self.cfg.seed,
                    regime="fast_drift" if self.cfg.regime == "fast" else self.cfg.regime,
                    n_agents=self.cfg.n_agents,
                ),
            )
        return out

    def _build_model(self, feat_dim: int) -> SECDO:
        m = SECDO(
            feat_dim=feat_dim,
            n_agents=self.cfg.n_agents,
            latent_dim=self.cfg.latent_dim,
            hidden=self.cfg.hidden,
            eta=self.cfg.eta,
            residual=True,
        )
        if self.cfg.load_predictor and Path(self.cfg.load_predictor).is_file():
            info = m.load_predictor_checkpoint(self.cfg.load_predictor)
            if self.ctx.is_main:
                print(f"Loaded predictor {self.cfg.load_predictor} missing={info['missing']}", flush=True)
        if self.cfg.resume and Path(self.cfg.resume).is_file():
            payload = load_checkpoint(self.cfg.resume, map_location="cpu")
            m.load_state_dict(payload["model"] if "model" in payload else payload, strict=False)
            if self.ctx.is_main:
                print(f"Resumed {self.cfg.resume}", flush=True)
        return m

    def run(self) -> list[dict]:
        print_banner(self.ctx, projection="anticipatory" if self.cfg.mode != "pretrain" else "n/a (pretrain)")
        if self.cfg.mode == "pretrain":
            return self.run_pretrain()
        if self.cfg.mode == "joint":
            return self.run_joint()
        if self.cfg.mode == "online":
            return self.run_online()
        raise ValueError(self.cfg.mode)

    # ─── Stage I ─────────────────────────────────────────────
    def run_pretrain(self) -> list[dict]:
        data_root = self._ensure_offline_data()
        ds = TrajectoryDataset(data_root)
        n_val = min(self.cfg.n_val_traj, max(1, len(ds) // 5))
        n_train = min(self.cfg.n_train_traj, len(ds) - n_val)
        train_ds = Subset(ds, list(range(n_train)))
        val_ds = Subset(ds, list(range(n_train, n_train + n_val)))
        feat_dim = int(ds[0]["feat"].shape[-1])
        model = wrap_ddp(self._build_model(feat_dim), self.ctx)
        raw = unwrap(model)
        self.model = raw  # type: ignore[assignment]
        opt = torch.optim.AdamW(raw.predictor.parameters(), lr=self.cfg.lr)
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.cfg.epochs, 1))
            if self.cfg.cosine
            else None
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch,
            shuffle=True,
            collate_fn=collate_trajectories,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch, shuffle=False, collate_fn=collate_trajectories
        )
        best = float("inf")
        for ep in range(self.cfg.epochs):
            raw.train()
            tlog = TheoryLog()
            for batch in train_loader:
                batch = to_device(batch, self.ctx.device)
                feat, c = batch["feat"], batch["c"]
                if c.dim() == 2:
                    c = c.unsqueeze(-1)
                T = feat.shape[0]
                h = None
                loss = torch.zeros((), device=self.ctx.device)
                with autocast("cuda", enabled=self.ctx.amp):
                    for t in range(T - 1):
                        out = raw.predictor.forward_step(feat[t], c[t], h)
                        h = out["h"]
                        loss = loss + predictor_loss(
                            out["s_hat"],
                            feat[t + 1],
                            out["c_hat"],
                            c[t + 1],
                            lambda_s=self.cfg.lambda_s,
                            lambda_c=self.cfg.lambda_c,
                            lambda_u=self.cfg.lambda_u,
                        )
                        with torch.no_grad():
                            tlog.eps.update(eps_l2(out["s_hat"], feat[t + 1]))
                            tlog.delta.update(float(constraint_loss(out["c_hat"].float(), c[t + 1])))
                            tlog.loss_c.update(float(constraint_loss(out["c_hat"].float(), c[t + 1])))
                            tlog.loss_s.update(float(dynamics_loss(out["s_hat"].float(), feat[t + 1])))
                    loss = loss / max(T - 1, 1)
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                tlog.loss_total.update(float(loss.detach()))
            if sched:
                sched.step()
            # val
            raw.eval()
            vlog = TheoryLog()
            with torch.no_grad():
                for batch in val_loader:
                    batch = to_device(batch, self.ctx.device)
                    feat, c = batch["feat"], batch["c"]
                    if c.dim() == 2:
                        c = c.unsqueeze(-1)
                    h = None
                    for t in range(feat.shape[0] - 1):
                        out = raw.predictor.forward_step(feat[t], c[t], h)
                        h = out["h"]
                        vlog.eps.update(eps_l2(out["s_hat"], feat[t + 1]))
                        vlog.delta.update(float(constraint_loss(out["c_hat"], c[t + 1])))
                        vlog.loss_c.update(float(constraint_loss(out["c_hat"], c[t + 1])))
            metrics = vlog.as_dict()
            metrics["epoch"] = ep
            metrics["stage"] = "pretrain"
            self.logs.append(metrics)
            is_best = metrics["delta"] < best
            if is_best:
                best = metrics["delta"]
            if self.ctx.is_main:
                print(
                    f"[pretrain] ep={ep} epsilon={metrics['epsilon']:.6f} "
                    f"delta={metrics['delta']:.6f} constraint_MSE={metrics['constraint_MSE']:.6f}",
                    flush=True,
                )
                save_best_last(
                    self.cfg.ckpt_dir,
                    {"epoch": ep, "model": raw.predictor.state_dict(), "metrics": metrics, "feat_dim": feat_dim},
                    is_best=is_best,
                )
                log_scalars(self.writer, "pretrain", metrics, ep)
        if self.ctx.is_main and self.logs:
            m = self.logs[-1]
            print_banner(self.ctx, projection="n/a (pretrain)", epsilon=m["epsilon"], delta=m["delta"])
        if self.writer:
            self.writer.close()
        return self.logs

    # ─── Stage II ────────────────────────────────────────────
    def run_joint(self) -> list[dict]:
        env_cfg = UAVBandwidthConfig(
            regime=self.cfg.regime,
            horizon=self.cfg.horizon,
            n_agents=self.cfg.n_agents,
            seed=self.cfg.seed,
        )
        env = UAVBandwidthEnv(env_cfg, device=self.ctx.device)
        obs = env.reset(batch=1)
        feat_dim = int(obs["feat"].shape[-1])
        if not self.cfg.load_predictor:
            self.cfg.load_predictor = "checkpoints/secdo_predictor/best.pt"
        model = wrap_ddp(self._build_model(feat_dim), self.ctx)
        raw: SECDO = unwrap(model)  # type: ignore[assignment]
        self.model = raw
        best_gap = float("inf")
        schedule = [
            ("frozen_predictor", self.cfg.frozen_epochs),
            ("head_only", self.cfg.head_only_epochs),
            ("joint", self.cfg.joint_epochs),
        ]
        global_ep = 0
        for stage, n_ep in schedule:
            self._set_trainable(raw, stage)
            params = [p for p in raw.parameters() if p.requires_grad]
            opt = (
                torch.optim.AdamW(
                    params, lr=self.cfg.lr_joint if stage == "joint" else self.cfg.lr
                )
                if params
                else None
            )
            for ep in range(n_ep):
                metrics = self._joint_epoch(env, raw, opt, stage)
                metrics.update({"epoch": ep, "stage": stage})
                self.logs.append(metrics)
                is_best = metrics["gap"] < best_gap
                if is_best:
                    best_gap = metrics["gap"]
                if self.ctx.is_main:
                    print(
                        f"[{stage}] ep={ep} epsilon={metrics['epsilon']:.6f} "
                        f"delta={metrics['delta']:.6f} violation={metrics['violation']:.6f} "
                        f"gap={metrics['gap']:.6f}",
                        flush=True,
                    )
                    save_best_last(
                        self.cfg.ckpt_dir,
                        {"stage": stage, "epoch": ep, "model": raw.predictor.state_dict(), "metrics": metrics},
                        is_best=is_best,
                    )
                    log_scalars(self.writer, stage, metrics, global_ep)
                global_ep += 1
        if self.ctx.is_main and self.logs:
            m = self.logs[-1]
            print_banner(
                self.ctx,
                projection="anticipatory",
                epsilon=m["epsilon"],
                delta=m["delta"],
                violation=m["violation"],
            )
        if self.writer:
            self.writer.close()
        return self.logs

    def _set_trainable(self, model: SECDO, stage: str) -> None:
        if stage == "frozen_predictor":
            for p in model.parameters():
                p.requires_grad = False
        elif stage == "head_only":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.constraint_head.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

    def _joint_epoch(self, env: UAVBandwidthEnv, model: SECDO, opt, stage: str) -> dict:
        n = env.cfg.n_agents
        tlog = TheoryLog()
        for _ in range(self.cfg.steps_per_epoch):
            obs = env.reset(batch=self.cfg.batch)
            pref = torch.ones(self.cfg.batch, n, device=self.ctx.device) / n
            x = reactive_project(pref.clone(), obs["c_teacher"])
            h = None
            L_pred = torch.zeros((), device=self.ctx.device)
            while not env.done:
                feat = to_device(obs["feat"], self.ctx.device)
                c_t = to_device(obs["c_teacher"], self.ctx.device)
                with autocast("cuda", enabled=self.ctx.amp):
                    out = model.forward_step(
                        x, pref, feat, c_t, h, mode="secdo", c_prev=None
                    )
                    h = out["h"]
                    x = out["x"]
                obs = env.step()
                c_next = to_device(obs["c_teacher"], self.ctx.device)
                with autocast("cuda", enabled=self.ctx.amp):
                    L_pred = L_pred + predictor_loss(
                        out["s_hat"],
                        obs["feat"].to(self.ctx.device),
                        out["c_hat"],
                        c_next,
                        lambda_s=self.cfg.lambda_s,
                        lambda_c=self.cfg.lambda_c,
                        lambda_u=self.cfg.lambda_u,
                    )
                with torch.no_grad():
                    tlog.eps.update(eps_l2(out["s_hat"], obs["feat"].to(self.ctx.device)))
                    tlog.delta.update(float((out["c_hat"].float() - c_next).abs().mean()))
                    tlog.rho.update(float((c_next - c_t).abs().mean()))
                    tlog.viol.update(float(violation(x, c_next)))
                    # gap vs projection of pref
                    from secdo.optimizer.anticipatory_projection import project_sum_budget

                    x_star = project_sum_budget(pref, c_next)
                    gap = float(
                        (allocation_objective(x, pref) - allocation_objective(x_star, pref))
                        .clamp(min=0)
                        .mean()
                    )
                    tlog.gap.update(gap)
            if opt is not None and stage != "frozen_predictor":
                opt.zero_grad(set_to_none=True)
                self.scaler.scale(L_pred).backward()
                self.scaler.step(opt)
                self.scaler.update()
            tlog.loss_total.update(float(L_pred.detach()) if torch.is_tensor(L_pred) else 0.0)
        return tlog.as_dict()

    # ─── Online ──────────────────────────────────────────────
    def run_online(self) -> list[dict]:
        env_cfg = UAVBandwidthConfig(
            regime=self.cfg.regime,
            horizon=max(self.cfg.horizon, self.cfg.online_steps),
            n_agents=self.cfg.n_agents,
            seed=self.cfg.seed,
        )
        env = UAVBandwidthEnv(env_cfg, device=self.ctx.device)
        obs = env.reset(batch=self.cfg.batch)
        feat_dim = int(obs["feat"].shape[-1])
        if not self.cfg.load_predictor:
            self.cfg.load_predictor = "checkpoints/secdo_joint/best.pt"
        model = self._build_model(feat_dim).to(self.ctx.device)
        self.model = model
        n = self.cfg.n_agents
        pref = torch.ones(self.cfg.batch, n, device=self.ctx.device) / n
        x = reactive_project(pref.clone(), obs["c_teacher"])
        h = None
        t = 0
        while t < self.cfg.online_steps and not env.done:
            feat = to_device(obs["feat"], self.ctx.device)
            c_t = to_device(obs["c_teacher"], self.ctx.device)
            out = model.forward_step(x, pref, feat, c_t, h, mode="anticipatory")
            h = out["h"].detach()
            x = out["x"]
            obs = env.step()
            c_next = to_device(obs["c_teacher"], self.ctx.device)
            # trust-region L_c on head
            before = {n_: p.detach().clone() for n_, p in model.constraint_head.named_parameters()}
            c_hat_g = model.constraint_head(out["h"].detach(), c_t)
            loss = constraint_loss(c_hat_g, c_next)
            model.constraint_head.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.constraint_head.parameters(), max_norm=self.cfg.zeta)
            with torch.no_grad():
                for p in model.constraint_head.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-self.cfg.lr)
                # ||Δθ|| ≤ ζ
                sq = sum(float((p - before[n_]).pow(2).sum()) for n_, p in model.constraint_head.named_parameters())
                norm = sq**0.5
                if norm > self.cfg.zeta and norm > 1e-12:
                    scale = self.cfg.zeta / norm
                    for n_, p in model.constraint_head.named_parameters():
                        p.copy_(before[n_] + (p - before[n_]) * scale)
            metrics = {
                "t": t,
                "stage": "online",
                "delta": float((out["c_hat"] - c_next).abs().mean()),
                "rho": float((c_next - c_t).abs().mean()),
                "violation": float(violation(x, c_next)),
                "L_c": float(loss.detach()),
            }
            self.logs.append(metrics)
            t += 1
        if self.ctx.is_main:
            save_best_last(
                self.cfg.ckpt_dir,
                {"model": model.predictor.state_dict(), "logs": self.logs},
                is_best=True,
            )
            if self.logs:
                print_banner(
                    self.ctx,
                    projection="anticipatory",
                    delta=self.logs[-1]["delta"],
                    violation=self.logs[-1]["violation"],
                )
                print(f"[online] last={self.logs[-1]}", flush=True)
        if self.writer:
            self.writer.close()
        return self.logs
