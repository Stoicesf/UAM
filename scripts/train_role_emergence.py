#!/usr/bin/env python3
"""Role emergence training (REINFORCE) — optional heterogeneous capability matching.

  python scripts/train_role_emergence.py --hetero --steps 50000 \\
      --save_dir experiment_results/hetero_training/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dice.local_obs import neighbor_mask
from dice.role_reward import RoleReward, role_entropy
from environments.dice_vmas_env import DICEVMASEnv, UAV_TYPES
from models.role_policy import RolePolicy


def light_scout_ratio(roles: torch.Tensor, uav_types: list[int]) -> float:
    """Fraction of light airframes currently assigned SCOUT."""
    light_idx = [i for i, t in enumerate(uav_types) if UAV_TYPES[t]["name"] == "light"]
    if not light_idx:
        return 0.0
    scouts = sum(1 for i in light_idx if int(roles[i]) == 0)
    return scouts / len(light_idx)


def train(
    *,
    steps: int = 50_000,
    n: int = 16,
    hetero: bool = False,
    hetero_ratio: tuple[float, float, float] = (0.3, 0.4, 0.3),
    match_anneal_steps: int = 30_000,
    max_steps: int = 80,
    save_dir: Path | None = None,
    seed: int = 0,
) -> dict:
    env = DICEVMASEnv(
        n_agents=n,
        n_tasks=5,
        max_steps=max_steps,
        n_roles=3,
        device="cpu",
        heterogeneous=hetero,
        hetero_ratio=hetero_ratio,
        hetero_role_bias=False,  # emergence: no hard seed
        seed=seed,
    )
    policy = RolePolicy(env.obs_dim, n_roles=3)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rr = RoleReward(3)
    hist: list[dict] = []
    global_step = 0
    ep = 0

    while global_step < steps:
        obs, info = env.reset(seed=seed + ep)
        logps: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        done = False
        while not done and global_step < steps:
            actions, _roles = policy(obs)
            logits = actions[:, 2:]
            dist = torch.distributions.Categorical(logits=logits)
            sampled = dist.sample()
            logp = dist.log_prob(sampled)
            actions = actions.clone()
            actions[:, 2:] = torch.nn.functional.one_hot(sampled, 3).float() * 2
            obs, r, done, trunc, info = env.step(actions)
            mask = neighbor_mask(env.pos, env.cfg.comm_radius, env.alive)
            # anneal match bonus: full until match_anneal_steps, then → 0
            match_w = max(0.0, 1.0 - global_step / max(match_anneal_steps, 1)) if hetero else 0.0
            shaped = r + 0.1 * rr.compute(
                info["roles"],
                info["prev_roles"],
                mask,
                uav_types=env.uav_type_list if hetero else None,
                match_weight=match_w,
            )
            logps.append(logp)
            rewards.append(shaped)
            global_step += 1
            if trunc or done:
                done = True

        G = torch.zeros(n)
        returns: list[torch.Tensor] = []
        for rw in reversed(rewards):
            G = rw + 0.95 * G
            returns.append(G)
        returns = list(reversed(returns))
        loss = torch.tensor(0.0)
        for lp, ret in zip(logps, returns):
            loss = loss - (lp * ret.detach()).mean()
        if logps:
            opt.zero_grad()
            loss.backward()
            opt.step()

        ent = role_entropy(info["roles"], 3)
        lsr = light_scout_ratio(info["roles"], env.uav_type_list) if hetero else 0.0
        row = {
            "ep": ep,
            "step": global_step,
            "entropy": ent,
            "coverage": info["coverage"],
            "loss": float(loss),
            "match_w": match_w if hetero else 0.0,
            "light_scout_ratio": lsr,
        }
        hist.append(row)
        if ep % 25 == 0 or global_step >= steps:
            print(
                f"ep={ep} step={global_step}/{steps} entropy={ent:.3f} "
                f"cov={info['coverage']:.2f} match_w={row['match_w']:.2f} light_scout={lsr:.2f}"
            )
        ep += 1

    out = {
        "final_entropy": hist[-1]["entropy"],
        "final_coverage": hist[-1]["coverage"],
        "final_light_scout_ratio": hist[-1].get("light_scout_ratio", 0.0),
        "steps": global_step,
        "hist": hist,
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "state_dict": policy.state_dict(),
            "obs_dim": env.obs_dim,
            "n_roles": 3,
            "hetero": hetero,
            "hetero_ratio": list(hetero_ratio),
            "steps": global_step,
        }
        torch.save(ckpt, save_dir / "role_policy.pt")
        (save_dir / "summary.json").write_text(
            json.dumps({k: v for k, v in out.items() if k != "hist"}, indent=2),
            encoding="utf-8",
        )
        (save_dir / "hist.json").write_text(json.dumps(hist[-500:], indent=2), encoding="utf-8")
        print(f"saved → {save_dir / 'role_policy.pt'}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=0, help="legacy; ignored if --steps set")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--hetero", action="store_true")
    ap.add_argument("--hetero_ratio", type=str, default="0.3,0.4,0.3")
    ap.add_argument("--match_anneal_steps", type=int, default=30_000)
    ap.add_argument("--save_dir", type=str, default="experiment_results/hetero_training")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.episodes > 0 and args.steps == 50_000:
        # legacy: ~80 steps/ep
        args.steps = args.episodes * 80

    parts = [float(x) for x in args.hetero_ratio.split(",")]
    if len(parts) != 3:
        raise SystemExit("--hetero_ratio needs 3 floats")
    ratio = (parts[0], parts[1], parts[2])
    save = Path(args.save_dir) if args.save_dir else None
    out = train(
        steps=args.steps,
        n=args.n,
        hetero=bool(args.hetero),
        hetero_ratio=ratio,
        match_anneal_steps=args.match_anneal_steps,
        save_dir=save,
        seed=args.seed,
    )
    print(
        f"train_role_emergence: PASS (entropy={out['final_entropy']:.3f}, "
        f"light_scout={out['final_light_scout_ratio']:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
