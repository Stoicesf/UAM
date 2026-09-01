"""Shared helpers for TRO evidence collectors."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from algorithms.baseline.mappo import build_mappo
from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from utils.experiment import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]


def resolve_device(device: str | None) -> str:
    req = (device or "auto").lower()
    if req in ("auto", "gpu"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            print("WARN: CUDA requested but unavailable; falling back to cpu", flush=True)
            return "cpu"
        return req
    return "cpu"


def spawn_kwargs(n: int) -> dict:
    if n >= 128:
        return {"world_spawning_x": 4.0, "world_spawning_y": 4.0}
    if n >= 64:
        return {"world_spawning_x": 3.0, "world_spawning_y": 3.0}
    if n >= 32:
        return {"world_spawning_x": 2.0, "world_spawning_y": 2.0}
    if n >= 16:
        return {"world_spawning_x": 1.5, "world_spawning_y": 1.5}
    return {}


def list_ac_dsgf_ckpts(seeds: list[int]) -> list[tuple[int, Path]]:
    found = []
    for s in seeds:
        p = ROOT / f"results/ac_dsgf/uav16/s{s}/checkpoints/final.pt"
        if p.exists():
            found.append((s, p))
    if not found:
        raise FileNotFoundError("No frozen uav16 final.pt for requested seeds")
    return found


def load_frozen_ac_dsgf(
    n: int,
    seed: int,
    ckpt_path: Path,
    device: str = "cpu",
    *,
    config_rel: str = "configs/ac_dsgf/ac_dsgf_16uav.yaml",
):
    cfg = load_experiment_config(str(ROOT / config_rel))
    cfg["env"]["num_agents"] = n
    cfg["env"]["num_envs"] = 1
    cfg["env"]["device"] = device
    cfg["env"].setdefault("max_steps", 128)
    cfg["train"]["seed"] = seed
    cfg["env"].update(spawn_kwargs(n))
    components = build_guided_mappo(cfg["train"], cfg["env"], cfg["guidance"])
    map_loc = device if device.startswith("cuda") else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=map_loc, weights_only=False)
    components.policy.load_state_dict(ckpt["policy"], strict=False)
    components.policy.to(device)
    components.policy.eval()
    return cfg, components.env, components.policy


def load_method(
    methods: dict,
    method: str,
    seed: int,
    device: str,
    *,
    n_agents: int | None = None,
):
    spec = methods[method]
    if len(spec) == 3:
        name, cfg_path, ckpt_tmpl = spec
        cls = None
    else:
        cls, name, cfg_path, ckpt_tmpl = spec

    cfg = load_experiment_config(str(ROOT / cfg_path))
    if n_agents is not None:
        cfg["env"]["num_agents"] = int(n_agents)
    cfg["env"]["num_envs"] = 1
    cfg["env"]["device"] = device
    cfg["env"].setdefault("max_steps", 128)
    cfg["train"]["seed"] = seed
    if n_agents is not None:
        cfg["env"].update(spawn_kwargs(n_agents))

    ckpt_path = ROOT / ckpt_tmpl.format(seed=seed)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    algo = cfg["train"].get("algorithm", "guided_mappo")
    if algo == "mappo" or cfg.get("guidance", {}).get("mode") == "none":
        components = build_mappo(cfg["train"], cfg["env"])
    else:
        components = build_guided_mappo(cfg["train"], cfg["env"], cfg.get("guidance", {}))

    map_loc = device if device.startswith("cuda") else "cpu"
    ckpt = torch.load(str(ckpt_path), map_location=map_loc, weights_only=False)
    incompat = components.policy.load_state_dict(ckpt["policy"], strict=False)
    print(
        f"  load {method} missing={len(incompat.missing_keys)} "
        f"unexpected={len(incompat.unexpected_keys)}",
        flush=True,
    )
    components.policy.to(device)
    components.policy.eval()
    if cls is None:
        return cfg, components.env, components.policy, name
    return cfg, components.env, components.policy, cls, name


def get_adapter(policy) -> ACGuideAdapter | None:
    for m in policy.modules():
        if isinstance(m, ACGuideAdapter):
            return m
    return None


def as_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().mean().item())
    return float(x)


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
