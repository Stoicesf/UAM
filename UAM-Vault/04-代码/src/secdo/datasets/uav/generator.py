"""Offline UAV trajectory generation + manifest (paper reproducibility)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from secdo.datasets.uav.channel import ChannelConfig
from secdo.datasets.uav.mobility import MobilityConfig, get_mobility
from secdo.datasets.uav.teacher import capacity_teacher, pack_state


@dataclass
class GenerateConfig:
    version: str = "v1"
    trajectories: int = 1000
    length: int = 200
    seed: int = 42
    regime: str = "fast_drift"
    n_agents: int = 8
    dim: int = 2
    dt: float = 0.12
    box: float = 3.0
    vel_scale: float = 0.4
    batch_size: int = 1  # stored as single-env trajectories
    device: str = "cpu"
    # stress: corrupt sensor views only (teacher stays clean)
    stress_eps: float = 0.0
    stress_delta: float = 0.0


def generate_trajectory(
    cfg: GenerateConfig,
    traj_id: int,
    channel: ChannelConfig | None = None,
) -> dict[str, torch.Tensor]:
    """One trajectory of length T; teacher c_t from capacity_teacher only."""
    device = torch.device(cfg.device)
    channel = channel or ChannelConfig(
        bandwidth_hz=1.0,
        comm_radius=3.5 if "slow" in cfg.regime else 2.8,
        path_loss_exp=2.2,
    )
    mob_cfg = MobilityConfig(
        n_agents=cfg.n_agents,
        dim=cfg.dim,
        dt=cfg.dt,
        box=cfg.box,
        vel_scale=cfg.vel_scale,
        seed=cfg.seed + traj_id,
    )
    mob = get_mobility(cfg.regime, mob_cfg)
    g = torch.Generator(device=device)
    g.manual_seed(cfg.seed + traj_id)
    pos, vel = mob.init_state(batch=1, device=device, generator=g)

    feats, cs, rhos, positions = [], [], [], []
    c_prev = None
    for t in range(cfg.length):
        pack = pack_state(pos, vel, cfg=channel)
        c = pack["c_teacher"]
        feat = pack["feat"]
        if cfg.stress_eps > 0:
            feat = feat + cfg.stress_eps * torch.randn_like(feat)
        rho = torch.zeros_like(c) if c_prev is None else (c - c_prev).abs()
        c_prev = c.detach().clone()

        # sanity: teacher must equal direct call
        c_direct = capacity_teacher(pos, channel)
        if not torch.allclose(c, c_direct, atol=1e-5, rtol=1e-5):
            raise RuntimeError("teacher inconsistency inside generator")

        feats.append(feat.squeeze(0).cpu())
        cs.append(c.squeeze(0).cpu())
        rhos.append(rho.squeeze(0).cpu())
        positions.append(pos.squeeze(0).cpu())

        if t + 1 < cfg.length:
            pos, vel = mob.step(pos, vel)

    return {
        "feat": torch.stack(feats, dim=0),  # [T, F]
        "c": torch.stack(cs, dim=0),  # [T, 1]
        "rho": torch.stack(rhos, dim=0),
        "position": torch.stack(positions, dim=0),  # [T, N, D]
        "traj_id": torch.tensor(traj_id),
    }


def write_manifest(out_dir: Path, cfg: GenerateConfig, extra: dict[str, Any] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": cfg.version,
        "trajectories": cfg.trajectories,
        "length": cfg.length,
        "seed": cfg.seed,
        "regime": cfg.regime,
        "n_agents": cfg.n_agents,
        "dt": cfg.dt,
        "box": cfg.box,
        "vel_scale": cfg.vel_scale,
        "teacher": "c_t = W log2(1 + SINR_bar)",
        "teacher_module": "secdo.datasets.uav.teacher.capacity_teacher",
    }
    if extra:
        manifest.update(extra)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def generate_dataset(
    out_dir: str | Path,
    cfg: GenerateConfig | None = None,
) -> Path:
    """
    Write:
      out_dir/manifest.json
      out_dir/trajectories.pt   # list[dict] or stacked
    """
    cfg = cfg or GenerateConfig()
    out_dir = Path(out_dir)
    samples = [generate_trajectory(cfg, i) for i in range(cfg.trajectories)]
    write_manifest(out_dir, cfg, extra={"n_written": len(samples)})
    torch.save({"trajectories": samples, "config": asdict(cfg)}, out_dir / "trajectories.pt")
    return out_dir


def _cfg_from_yaml(path: str | Path) -> tuple[GenerateConfig, Path]:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = Path(raw.pop("out_dir", "data/secdo/uav_v1"))
    known = {f.name for f in GenerateConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in raw.items() if k in known}
    return GenerateConfig(**kwargs), out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Generate SECDO UAV offline dataset")
    p.add_argument("--config", type=str, default="secdo/configs/data/uav_fast.yaml")
    p.add_argument("--quick", action="store_true", help="2 traj × 16 steps smoke dump")
    args = p.parse_args(argv)
    cfg, out = _cfg_from_yaml(args.config)
    if args.quick:
        cfg.trajectories = 2
        cfg.length = 16
        out = Path(str(out) + "_quick")
    path = generate_dataset(out, cfg)
    print(f"Wrote dataset → {path}", flush=True)
    print((path / "manifest.json").read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
