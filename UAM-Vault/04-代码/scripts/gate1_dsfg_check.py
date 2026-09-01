"""Gate 1 — DSGF forward chain: obs -> DSGF -> Phi -> NavRL -> action."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from algorithms.guided.mappo_guided import build_guided_mappo
from models.dsgf import DSGF
from reward.guidance_reward import guidance_alignment_reward
from utils.seed import set_seed


def load_cfg(path: str) -> tuple[dict, dict, dict]:
    with open(path, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    with open(ROOT / "configs/train.yaml", encoding="utf-8") as f:
        base_train = yaml.safe_load(f) or {}
    with open(ROOT / "configs/environment.yaml", encoding="utf-8") as f:
        base_env = yaml.safe_load(f) or {}
    return (
        {**base_train, **exp.get("train", {})},
        {**base_env, **exp.get("env", {})},
        exp.get("guidance", {}),
    )


def check_dsfg_standalone(obs_dim: int = 18, n_agents: int = 4):
    enc = DSGF(obs_dim, hidden_dim=128, guidance_dim=6, comm_radius=0.5)
    obs = torch.randn(2, n_agents, obs_dim)
    phi = enc(obs)
    assert phi.shape == (2, n_agents, 6), f"phi shape {phi.shape}"
    assert not torch.isnan(phi).any(), "phi contains NaN"
    assert phi.abs().mean() > 1e-6, "phi collapsed to zero"
    print(f"[OK] DSGF standalone: phi {tuple(phi.shape)}, mean={phi.mean():.4f}")


def check_guided_policy(train_cfg, env_cfg, guidance_cfg):
    components = build_guided_mappo(train_cfg, env_cfg, guidance_cfg)
    env = components.env
    td = env.reset()

    obs = td.get(("agents", "observation"))
    obs_dim = obs.shape[-1]
    phi_dim = guidance_cfg.get("guidance_dim", 6)
    print(f"obs.shape = {tuple(obs.shape)}")

    with torch.no_grad():
        components.policy(td)

    phi = td.get(("agents", "phi"))
    assert phi is not None, "phi not set by policy forward"
    print(f"phi.shape = {tuple(phi.shape)}")
    print(f"phi.mean() = {phi.mean().item():.4f}")
    print(f"phi.std()  = {phi.std().item():.4f}")
    assert phi.shape[-1] == phi_dim, f"expected phi dim {phi_dim}, got {phi.shape[-1]}"
    assert not torch.isnan(phi).any(), "phi is NaN"

    aug_dim = obs_dim + phi_dim
    print(f"Actor input dim (obs + phi) = {aug_dim}")

    td = env.rollout(max_steps=1, policy=components.policy, auto_cast_to_device=True)
    actions = td.get(("agents", "action"))
    phi_r = td.get(("agents", "phi"))
    align = guidance_alignment_reward(actions, phi_r)
    print(f"action_alignment (1-step) = {align.mean().item():.4f}")
    print("[OK] Gate 1 passed: obs -> DSGF -> Phi -> NavRL -> action")


def main():
    parser = argparse.ArgumentParser(description="DSGF Gate 1 forward check")
    parser.add_argument("--exp", default="configs/experiments/exp4_dsfg.yaml")
    args = parser.parse_args()

    train_cfg, env_cfg, guidance_cfg = load_cfg(args.exp)
    set_seed(train_cfg.get("seed", 42))

    print("=== DSGF Gate 1: Forward Chain Validation ===")
    check_dsfg_standalone(env_cfg.get("obs_dim", 18), env_cfg.get("num_agents", 4))
    try:
        check_guided_policy(train_cfg, env_cfg, guidance_cfg)
    except Exception as e:
        print(f"[FAIL] Gate 1: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
