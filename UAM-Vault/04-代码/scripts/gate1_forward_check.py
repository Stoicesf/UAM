"""Gate 1 — Forward & reward wiring validation (~5 min)."""

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
from guidance.guide_encoder import GuideEncoder
from reward.guidance_reward import guidance_alignment_reward
from utils.seed import set_seed


def load_cfg(path: str) -> tuple[dict, dict, dict]:
    with open(path, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    with open("configs/train.yaml", encoding="utf-8") as f:
        base_train = yaml.safe_load(f) or {}
    with open("configs/environment.yaml", encoding="utf-8") as f:
        base_env = yaml.safe_load(f) or {}
    train = {**base_train, **exp.get("train", {})}
    env = {**base_env, **exp.get("env", {})}
    guidance = exp.get("guidance", {})
    return train, env, guidance


def check_guide_encoder_standalone(obs_dim: int = 18):
    enc = GuideEncoder(obs_dim, hidden_dim=128, guidance_dim=4)
    obs = torch.randn(2, 4, obs_dim)
    phi = enc(obs)
    assert phi.shape == (2, 4, 4), f"phi shape {phi.shape}"
    assert not torch.isnan(phi).any(), "phi contains NaN"
    assert phi.abs().mean() > 1e-6, "phi collapsed to zero"
    print(f"[OK] GuideEncoder standalone: phi {tuple(phi.shape)}, mean={phi.mean():.4f}, std={phi.std():.4f}")


def check_guided_policy(train_cfg, env_cfg, guidance_cfg):
    components = build_guided_mappo(train_cfg, env_cfg, guidance_cfg)
    env = components.env
    td = env.reset()

    obs = td.get(("agents", "observation"))
    obs_dim = obs.shape[-1]
    print(f"obs.shape = {tuple(obs.shape)}")

    with torch.no_grad():
        components.policy(td)

    phi = td.get(("agents", "phi"))
    assert phi is not None, "phi not set by policy forward"
    print(f"phi.shape = {tuple(phi.shape)}")
    print(f"phi.mean() = {phi.mean().item():.4f}")
    print(f"phi.std()  = {phi.std().item():.4f}")
    assert not torch.isnan(phi).any(), "phi is NaN"
    assert phi.abs().mean() > 1e-6, "phi is all zero"

    aug_dim = obs_dim + guidance_cfg.get("guidance_dim", 4)
    print(f"Actor input dim (obs + phi) = {aug_dim}")

    # Rollout 1 step
    td = env.rollout(max_steps=1, policy=components.policy, auto_cast_to_device=True)
    actions = td.get(("agents", "action"))
    phi_r = td.get(("agents", "phi"))
    align = guidance_alignment_reward(actions, phi_r)
    print(f"action_alignment (1-step rollout) = {align.mean().item():.4f}")

    info = td.get(("next", "agents", "info"))
    if info is not None:
        print(f"VMAS info keys: {list(info.keys())}")
        for k in ("pos_rew", "final_rew", "agent_collisions"):
            if k in info.keys():
                print(f"  {k}.mean = {info.get(k).mean().item():.4f}")

    print("[OK] Gate 1 passed: Guide participates in policy forward.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Stage 1 Gate 1 forward check")
    parser.add_argument("--exp", default="configs/experiments/exp1_guide.yaml")
    args = parser.parse_args()

    train_cfg, env_cfg, guidance_cfg = load_cfg(args.exp)
    set_seed(train_cfg.get("seed", 42))

    print("=== Gate 1: Forward & Reward Validation ===")
    check_guide_encoder_standalone(env_cfg.get("obs_dim", 18))
    try:
        check_guided_policy(train_cfg, env_cfg, guidance_cfg)
    except Exception as e:
        print(f"[FAIL] Gate 1: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
