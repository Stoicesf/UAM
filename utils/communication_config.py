"""P1-3 — merge method + communication radius experiment configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from utils.experiment import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "communication" / "manifest.json"
GENERATED_DIR = ROOT / "configs" / "communication" / "generated"


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_radius_spec(radius_key: str) -> dict:
    manifest = load_manifest()
    path = ROOT / manifest["radius_configs"][radius_key]
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_comm_experiment_config(method: str, radius_key: str) -> dict:
    """Merge frozen method config with communication radius override."""
    manifest = load_manifest()
    if method not in manifest["methods"]:
        raise ValueError(f"Unknown method: {method}")
    if radius_key not in manifest["radius_configs"]:
        raise ValueError(f"Unknown radius key: {radius_key}")

    method_cfg = load_experiment_config(str(ROOT / manifest["methods"][method]["exp_config"]))
    radius_spec = load_radius_spec(radius_key)
    comm_r = float(radius_spec["comm_radius"])
    label = radius_spec.get("label", radius_key)

    cfg = copy.deepcopy(method_cfg)
    cfg["env"]["comm_radius"] = comm_r
    guidance = cfg.setdefault("guidance", {})
    if guidance.get("mode", "none") != "none":
        guidance["comm_radius"] = comm_r

    exp = cfg.setdefault("experiment", {})
    exp["id"] = f"comm_{method}_r{label}"
    exp["name"] = f"{method.upper()} @ R={label} (16 UAV)"

    cfg.setdefault("output", {})["category"] = "communication"
    cfg["comm_sweep"] = {
        "method": method,
        "radius_key": radius_key,
        "radius_label": label,
        "comm_radius": comm_r,
    }
    return cfg


def write_generated_config(method: str, radius_key: str, run_name: str) -> Path:
    """Write merged YAML for train.py --exp."""
    cfg = build_comm_experiment_config(method, radius_key)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out = GENERATED_DIR / f"{run_name}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return out


def list_runs(profile: str = "full") -> list[dict[str, str]]:
    manifest = load_manifest()
    if profile not in manifest["profiles"]:
        raise ValueError(f"Unknown profile: {profile}. Choose from {list(manifest['profiles'])}")
    return manifest["profiles"][profile]
