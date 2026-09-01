"""Paper-level Demo: checkpoint → VMAS rollout → trajectory + graph frames + video.

Recommended (Day 1):
  python scripts/demo_runner.py --method dsgf --ckpt-prefer 20k --episodes 8 --select-best

Comparison (Day 2, if MAPPO ckpt exists):
  python scripts/demo_runner.py --method mappo --ckpt results/.../final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed
from visualization.draw_graph import adjacency_from_positions, draw_frame, plot_trajectories
from visualization.render_frames import frames_to_gif, frames_to_mp4

DEFAULT_CKPTS = {
    "dsgf": {
        "20k": "results/dsgf/dsfg_v2_102k/checkpoints/checkpoint_20k.pt",
        "final": "results/dsgf/dsfg_v2_102k/checkpoints/final.pt",
        "exp": "configs/demo/dsgf_4uav.yaml",
    },
    "gat": {
        "20k": "results/graph/gat_a_lambda003/checkpoints/checkpoint_20k.pt",
        "final": "results/graph/gat_a_lambda003/checkpoints/final.pt",
        "exp": "configs/demo/gat_4uav.yaml",
    },
    "mappo": {
        "20k": "results/demo/mappo_4uav/checkpoints/checkpoint_20k.pt",
        "final": "results/demo/mappo_4uav/checkpoints/final.pt",
        "exp": "configs/demo/mappo_4uav.yaml",
    },
}


def _obs_positions(obs: torch.Tensor) -> np.ndarray:
    """obs: (B, N, d) or (N, d) → (N, 2)."""
    if obs.dim() == 3:
        obs = obs[0]
    return obs[:, :2].detach().cpu().numpy()


def _obs_goals(obs: torch.Tensor) -> np.ndarray:
    """Absolute goals from pos + goal_rel (nav obs indices 4:6)."""
    if obs.dim() == 3:
        obs = obs[0]
    pos = obs[:, :2]
    goal_rel = obs[:, 4:6]
    return (pos + goal_rel).detach().cpu().numpy()


def _success_rate(obs: torch.Tensor, threshold: float = 0.3) -> float:
    if obs.dim() == 3:
        obs = obs[0]
    dist = obs[:, 4:6].norm(dim=-1)
    return (dist < threshold).float().mean().item()


def rollout_episode(
    env,
    policy,
    max_steps: int,
    comm_radius: float,
    record_frames: bool,
    frame_dir: Path | None,
    title_prefix: str,
) -> dict:
    policy.eval()
    trajs: list[list[tuple[float, float]]] = []
    adjs: list[np.ndarray] = []
    goals_final = None
    success_t = []

    with torch.no_grad():
        td = env.reset()
        obs = td.get(("agents", "observation"))
        pos = _obs_positions(obs)
        goals = _obs_goals(obs)
        goals_final = goals.copy()
        n = pos.shape[0]
        trajs = [[(float(pos[i, 0]), float(pos[i, 1]))] for i in range(n)]
        adj = adjacency_from_positions(pos, comm_radius)
        adjs.append(adj)

        xlim = (float(pos[:, 0].min() - 0.5), float(pos[:, 0].max() + 0.5))
        ylim = (float(pos[:, 1].min() - 0.5), float(pos[:, 1].max() + 0.5))
        # Expand bounds using goals
        xlim = (min(xlim[0], float(goals[:, 0].min()) - 0.3), max(xlim[1], float(goals[:, 0].max()) + 0.3))
        ylim = (min(ylim[0], float(goals[:, 1].min()) - 0.3), max(ylim[1], float(goals[:, 1].max()) + 0.3))

        if record_frames and frame_dir is not None:
            frame_dir.mkdir(parents=True, exist_ok=True)
            draw_frame(
                pos, goals, adj,
                out_path=frame_dir / "frame_0000.png",
                title=f"{title_prefix} t=0",
                xlim=xlim, ylim=ylim,
            )

        for t in range(1, max_steps + 1):
            td = policy(td)
            td = env.step(td)
            obs = td.get(("next", "agents", "observation"))
            pos = _obs_positions(obs)
            goals = _obs_goals(obs)
            goals_final = goals.copy()
            for i in range(n):
                trajs[i].append((float(pos[i, 0]), float(pos[i, 1])))
            adj = adjacency_from_positions(pos, comm_radius)
            adjs.append(adj)
            s = _success_rate(obs)
            success_t.append(s)

            if record_frames and frame_dir is not None:
                draw_frame(
                    pos, goals, adj,
                    out_path=frame_dir / f"frame_{t:04d}.png",
                    title=f"{title_prefix} t={t}  S={s:.0%}",
                    xlim=xlim, ylim=ylim,
                )

            done = td.get(("next", "done"))
            if done is not None and bool(done.any()):
                break
            td = step_mdp(td)

    arrays = [np.asarray(tr, dtype=np.float32) for tr in trajs]
    final_s = success_t[-1] if success_t else 0.0
    mean_edges = float(np.mean([a.sum() for a in adjs])) if adjs else 0.0
    return {
        "trajectories": arrays,
        "goals": goals_final,
        "adjacencies": adjs,
        "final_success": final_s,
        "mean_edges": mean_edges,
        "steps": len(arrays[0]) if arrays else 0,
    }


def resolve_demo_ckpt(method: str, prefer: str, explicit: str | None) -> Path:
    if explicit:
        return resolve_checkpoint(explicit)
    spec = DEFAULT_CKPTS[method]
    primary = spec.get(prefer) or spec["final"]
    fallback = spec["final"] if prefer == "20k" else spec.get("20k")
    return resolve_checkpoint(primary, fallback)


def main():
    parser = argparse.ArgumentParser(description="Paper-level DSGF/MAPPO demo")
    parser.add_argument("--method", choices=["dsgf", "mappo", "gat"], default="dsgf")
    parser.add_argument("--exp", default=None, help="Override experiment YAML")
    parser.add_argument("--ckpt", default=None, help="Checkpoint path")
    parser.add_argument("--ckpt-prefer", choices=["20k", "final"], default="20k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--select-best", action="store_true", help="Keep episode with max final success")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    exp_path = args.exp or DEFAULT_CKPTS[args.method]["exp"]
    exp_cfg = load_experiment_config(str(ROOT / exp_path))
    exp_cfg["env"]["num_envs"] = 1
    exp_cfg["env"]["device"] = "cpu"
    if args.seed is not None:
        exp_cfg["train"]["seed"] = args.seed
    set_seed(args.seed)

    ckpt = resolve_demo_ckpt(args.method, args.ckpt_prefer, args.ckpt)
    print(f"Method: {args.method}")
    print(f"Config: {exp_path}")
    print(f"Checkpoint: {ckpt}")

    env, policy = load_policy_for_eval(exp_cfg, ckpt)
    comm_radius = float(
        exp_cfg.get("guidance", {}).get("comm_radius", exp_cfg["env"].get("comm_radius", 0.5))
    )
    success_th = float(exp_cfg["env"].get("success_threshold", 0.3))

    out_root = Path(args.out_dir) if args.out_dir else ROOT / "demo"
    tag = f"{args.method}_4uav"
    run_dir = out_root / tag
    frames_dir = run_dir / "frames"
    videos_dir = out_root / "videos"
    figures_dir = out_root / "figures"
    for d in (run_dir, frames_dir, videos_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Clear old frames
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()

    best = None
    history = []
    n_eps = args.episodes
    for ep in range(n_eps):
        # re-seed lightly for diversity when hunting best episode
        set_seed(args.seed + ep)
        tmp_frames = frames_dir / f"_ep{ep}"
        result = rollout_episode(
            env, policy,
            max_steps=args.max_steps,
            comm_radius=comm_radius,
            record_frames=True,
            frame_dir=tmp_frames,
            title_prefix=args.method.upper(),
        )
        history.append({
            "episode": ep,
            "final_success": result["final_success"],
            "mean_edges": result["mean_edges"],
            "steps": result["steps"],
        })
        print(
            f"  ep{ep}: success={result['final_success']:.2%} "
            f"edges={result['mean_edges']:.1f} steps={result['steps']}"
        )
        if best is None or result["final_success"] > best["final_success"]:
            best = result
            best["episode"] = ep
            best_frame_src = tmp_frames

        if not args.select_best:
            best = result
            best["episode"] = ep
            best_frame_src = tmp_frames
            break

    assert best is not None

    # Promote best frames to main frames_dir
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()
    for src in sorted(best_frame_src.glob("frame_*.png")):
        dest = frames_dir / src.name
        dest.write_bytes(src.read_bytes())

    # Cleanup episode dirs
    for p in frames_dir.glob("_ep*"):
        if p.is_dir():
            for f in p.glob("*"):
                f.unlink()
            p.rmdir()

    traj_path = figures_dir / f"trajectory_{args.method}.png"
    plot_trajectories(
        best["trajectories"],
        best["goals"],
        traj_path,
        title=f"{args.method.upper()} 4-UAV Trajectory (ep{best['episode']}, S={best['final_success']:.0%})",
    )
    print(f"Saved {traj_path}")

    # Mid-episode graph snapshot
    mid = best["adjacencies"][len(best["adjacencies"]) // 2]
    mid_pos = best["trajectories"][0]  # wrong - need positions at mid
    # Reconstruct mid positions from trajectories
    mid_t = len(best["trajectories"][0]) // 2
    mid_positions = np.stack([tr[mid_t] for tr in best["trajectories"]], axis=0)
    graph_path = figures_dir / f"graph_attention_{args.method}.png"
    draw_frame(
        mid_positions,
        best["goals"],
        mid,
        out_path=graph_path,
        title=f"{args.method.upper()} communication graph (mid episode)",
    )
    print(f"Saved {graph_path}")

    video_path = None
    if not args.no_video:
        video_path = videos_dir / f"{args.method}_4uav.mp4"
        out_media = frames_to_mp4(frames_dir, video_path, fps=args.fps)
        print(f"Saved {out_media}")
        if out_media.suffix == ".gif":
            # also keep gif name for graph attention if mp4 unavailable
            pass
        # Extra gif for paper embeds
        gif_path = videos_dir / f"{args.method}_4uav.gif"
        try:
            frames_to_gif(frames_dir, gif_path, fps=max(6, args.fps // 2))
            print(f"Saved {gif_path}")
        except Exception as e:
            print(f"[warn] gif failed: {e}")

    meta = {
        "method": args.method,
        "checkpoint": str(ckpt),
        "seed": args.seed,
        "best_episode": best["episode"],
        "final_success": best["final_success"],
        "mean_edges": best["mean_edges"],
        "steps": best["steps"],
        "comm_radius": comm_radius,
        "success_threshold": success_th,
        "history": history,
        "trajectory": str(traj_path),
        "graph": str(graph_path),
        "video": str(video_path) if video_path else None,
    }
    meta_path = run_dir / "metrics.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")
    print(
        f"[OK] Demo done: best S={best['final_success']:.2%} "
        f"(ep{best['episode']}/{n_eps})"
    )


if __name__ == "__main__":
    main()
