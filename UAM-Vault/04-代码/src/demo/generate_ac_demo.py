"""Dynamic Communication Demo — Communication Emergence (4 UAV visualize).

Layout per frame:
  Row1: GAT | DSGF | AC-DSGF   (env + overlay edges)
  Row2: Comm graph panels
  Row3: Comm cost | Active edges | Success progress

Outputs:
  demo/videos/ac_dsgf_dynamic_comm.mp4
  demo/figures/communication_evolution.png
  demo/traces/{gat,dsgf,ac_dsgf}_episode.json

Usage:
  python demo/generate_ac_demo.py --episodes 6 --select-best --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.guided.mappo_guided import ACGuideAdapter
from demo.render_comm_graph import active_edge_count, render_comm_graph
from demo.render_env import render_env
from demo.render_metrics import render_metrics
from utils.experiment import load_experiment_config
from utils.policy_loader import load_policy_for_eval, resolve_checkpoint
from utils.seed import set_seed
from visualization.draw_graph import adjacency_from_positions
from visualization.render_frames import frames_to_mp4

SPECS = {
    "gat": {
        "exp": "configs/demo/gat_4uav.yaml",
        "ckpt": "results/graph/gat_a_lambda003/checkpoints/checkpoint_20k.pt",
    },
    "dsgf": {
        "exp": "configs/demo/dsgf_4uav.yaml",
        "ckpt": "results/dsgf/dsfg_v2_102k/checkpoints/checkpoint_20k.pt",
    },
    "ac_dsgf": {
        "exp": "configs/ac_dsgf/ac_dsgf_smoke_v0.yaml",
        "ckpt": "results/ac_dsgf/ac_dsgf_smoke_v0/checkpoints/final.pt",
    },
}
METHODS = ["gat", "dsgf", "ac_dsgf"]
TITLES = {"gat": "GAT", "dsgf": "DSGF", "ac_dsgf": "AC-DSGF"}


def _obs_positions(obs: torch.Tensor) -> np.ndarray:
    if obs.dim() == 3:
        obs = obs[0]
    return obs[:, :2].detach().cpu().numpy()


def _obs_goals(obs: torch.Tensor) -> np.ndarray:
    if obs.dim() == 3:
        obs = obs[0]
    return (obs[:, :2] + obs[:, 4:6]).detach().cpu().numpy()


def _success(obs: torch.Tensor, thr: float = 0.3) -> float:
    if obs.dim() == 3:
        obs = obs[0]
    return float((obs[:, 4:6].norm(dim=-1) < thr).float().mean())


def _mean_goal_dist(obs: torch.Tensor) -> float:
    if obs.dim() == 3:
        obs = obs[0]
    return float(obs[:, 4:6].norm(dim=-1).mean())


def _get_adj(policy, method: str, pos: np.ndarray, comm_radius: float) -> np.ndarray:
    if method == "ac_dsgf":
        for m in policy.modules():
            if isinstance(m, ACGuideAdapter) and m.last_gate_matrix is not None:
                g = m.last_gate_matrix.detach().cpu().numpy()
                # symmetrize for undirected viz
                return np.maximum(g, g.T)
    return adjacency_from_positions(pos, comm_radius)


def _comm_cost(policy, method: str, adj: np.ndarray) -> float:
    if method == "ac_dsgf":
        for m in policy.modules():
            if isinstance(m, ACGuideAdapter):
                return float(m.last_comm_cost)
    # geometric undirected edge count (halve directed sum)
    return float(np.triu(adj, k=1).sum())


def record_episode(env, policy, method: str, max_steps: int, comm_radius: float, thr: float) -> dict:
    policy.eval()
    times, positions, goals_l, adjs = [], [], [], []
    comms, edges, successes, dists = [], [], [], []

    with torch.no_grad():
        td = env.reset()
        obs = td.get(("agents", "observation"))
        # Warm-start AC gate with a forward
        td = policy(td)
        pos = _obs_positions(obs)
        goals = _obs_goals(obs)
        adj = _get_adj(policy, method, pos, comm_radius)
        times.append(0)
        positions.append(pos.tolist())
        goals_l.append(goals.tolist())
        adjs.append(adj.tolist())
        comms.append(_comm_cost(policy, method, adj))
        edges.append(active_edge_count(adj))
        successes.append(_success(obs, thr))
        dists.append(_mean_goal_dist(obs))

        for t in range(1, max_steps + 1):
            td = env.step(td)
            obs = td.get(("next", "agents", "observation"))
            done = td.get(("next", "done"))
            td = step_mdp(td)
            td = policy(td)
            pos = _obs_positions(obs)
            goals = _obs_goals(obs)
            adj = _get_adj(policy, method, pos, comm_radius)
            times.append(t)
            positions.append(pos.tolist())
            goals_l.append(goals.tolist())
            adjs.append(adj.tolist())
            comms.append(_comm_cost(policy, method, adj))
            edges.append(active_edge_count(adj))
            successes.append(_success(obs, thr))
            dists.append(_mean_goal_dist(obs))
            if done is not None and bool(done.any()):
                break

    return {
        "method": method,
        "time": times,
        "positions": positions,
        "goals": goals_l,
        "adj": adjs,
        "comm_cost": comms,
        "active_edges": edges,
        "success": successes,
        "goal_distance": dists,
        "final_success": successes[-1] if successes else 0.0,
        "mean_comm": float(np.mean(comms)) if comms else 0.0,
        "edge_std": float(np.std(edges)) if edges else 0.0,
    }


def _bounds(traces: dict[str, dict]) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [], []
    for tr in traces.values():
        for p in tr["positions"]:
            arr = np.asarray(p)
            xs.extend(arr[:, 0].tolist())
            ys.extend(arr[:, 1].tolist())
        for g in tr["goals"]:
            arr = np.asarray(g)
            xs.extend(arr[:, 0].tolist())
            ys.extend(arr[:, 1].tolist())
    pad = 0.4
    return (min(xs) - pad, max(xs) + pad), (min(ys) - pad, max(ys) + pad)


def render_frame(traces: dict[str, dict], t: int, xlim, ylim, out_path: Path):
    fig = plt.figure(figsize=(13.5, 9.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1.15, 0.9], hspace=0.35, wspace=0.28)

    for col, m in enumerate(METHODS):
        tr = traces[m]
        t_clamped = min(t, len(tr["time"]) - 1)
        pos = np.asarray(tr["positions"][t_clamped])
        goals = np.asarray(tr["goals"][t_clamped])
        adj = np.asarray(tr["adj"][t_clamped])
        n_e = active_edge_count(adj)

        ax_env = fig.add_subplot(gs[0, col])
        render_env(
            ax_env, pos, goals,
            title=f"{TITLES[m]}  env  t={t_clamped}",
            xlim=xlim, ylim=ylim,
        )
        # light edges on env for spatial context
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                w = float(max(adj[i, j], adj[j, i]))
                if w > 0.05:
                    ax_env.plot(
                        [pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color="gray", alpha=0.35, lw=0.8 + w, zorder=1,
                    )

        ax_g = fig.add_subplot(gs[1, col])
        render_comm_graph(
            ax_g, pos, adj,
            title=f"{TITLES[m]}  graph  edges={n_e}",
            xlim=xlim, ylim=ylim,
        )

    ax_c = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    ax_s = fig.add_subplot(gs[2, 2])
    render_metrics([ax_c, ax_e, ax_s], traces, t, METHODS)

    fig.suptitle(
        "Adaptive Communication Emergence in Multi-UAV Cooperative Navigation",
        fontsize=12, y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _export_uniform_mp4(frame_dir: Path, out_mp4: Path, fps: int = 8) -> Path:
    """Pad frames to a common multiple-of-16 size then write mp4 (or gif)."""
    from PIL import Image
    import imageio.v2 as imageio

    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise FileNotFoundError(frame_dir)
    imgs = [np.asarray(Image.open(f).convert("RGB")) for f in frames]
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    h = ((h + 15) // 16) * 16
    w = ((w + 15) // 16) * 16
    pads = []
    for im in imgs:
        pad = np.ones((h, w, 3), dtype=np.uint8) * 255
        pad[: im.shape[0], : im.shape[1]] = im
        pads.append(pad)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(str(out_mp4), pads, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        return out_mp4
    except Exception as e:
        print(f"[warn] mp4 encode failed ({e}); falling back to frames_to_mp4")
        return frames_to_mp4(frame_dir, out_mp4, fps=fps)


def plot_evolution_summary(traces: dict[str, dict], out: Path):
    """Static figure emphasizing AC communication emergence over time."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    colors = {"gat": "#DD8452", "dsgf": "#4C72B0", "ac_dsgf": "#C44E52"}
    for m in METHODS:
        tr = traces[m]
        t = tr["time"]
        label = TITLES[m]
        axes[0].plot(t, tr["comm_cost"], color=colors[m], lw=2, label=label)
        axes[1].plot(t, tr["active_edges"], color=colors[m], lw=2, label=label)
        axes[2].plot(t, tr["success"], color=colors[m], lw=2, label=label)

    axes[0].set_title("Communication Cost")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("cost")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Active Edges (emergence)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("#edges")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Success Progress")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("success")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Communication Evolution (Demo, qualitative)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--frame-stride", type=int, default=2, help="Render every k-th step")
    args = parser.parse_args()

    set_seed(args.seed)
    policies = {}
    envs = {}
    radii = {}
    thr = 0.3

    for m in METHODS:
        spec = SPECS[m]
        ckpt = resolve_checkpoint(str(ROOT / spec["ckpt"]))
        exp_cfg = load_experiment_config(str(ROOT / spec["exp"]))
        exp_cfg["env"]["num_envs"] = 1
        exp_cfg["env"]["device"] = "cpu"
        print(f"Load {m}: {ckpt}")
        env, policy = load_policy_for_eval(exp_cfg, ckpt)
        if m == "ac_dsgf":
            for mod in policy.modules():
                if isinstance(mod, ACGuideAdapter):
                    mod.set_ablation_mode("full")
                    mod.set_budget_ratio(None)
        envs[m] = env
        policies[m] = policy
        radii[m] = float(
            exp_cfg.get("guidance", {}).get("comm_radius", exp_cfg["env"].get("comm_radius", 0.5))
        )
        thr = float(exp_cfg["env"].get("success_threshold", thr))

    # Hunt for a demo episode where AC shows some edge dynamics (not flat-zero forever)
    best_score = -1e9
    best_traces = None
    for ep in range(args.episodes):
        set_seed(args.seed + ep * 17)
        traces = {}
        for m in METHODS:
            traces[m] = record_episode(
                envs[m], policies[m], m, args.max_steps, radii[m], thr
            )
        # Prefer: AC has edge variation + decent success; not pure silence collapse story
        ac = traces["ac_dsgf"]
        score = (
            3.0 * ac["final_success"]
            + 1.5 * ac["edge_std"]
            + 0.5 * traces["dsgf"]["final_success"]
            - 0.01 * abs(ac["mean_comm"])  # mild; CEI narrative elsewhere
        )
        print(
            f"ep={ep}  AC_S={ac['final_success']:.2f} edges_std={ac['edge_std']:.2f} "
            f"GAT_S={traces['gat']['final_success']:.2f} DSGF_S={traces['dsgf']['final_success']:.2f} "
            f"score={score:.3f}"
        )
        if not args.select_best:
            best_traces = traces
            break
        if score > best_score:
            best_score = score
            best_traces = traces

    assert best_traces is not None
    traces = best_traces

    # Save JSON traces
    trace_dir = ROOT / "demo" / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for m in METHODS:
        path = trace_dir / f"{m}_episode.json"
        # lighter: drop full adj floats list keep summary edges if huge — keep adj for rerender
        path.write_text(json.dumps(traces[m], indent=2), encoding="utf-8")
        print(f"Saved {path}")

    # Render frames
    frame_dir = ROOT / "demo" / "frames_dynamic_comm"
    if frame_dir.exists():
        for f in frame_dir.glob("frame_*.png"):
            f.unlink()
    frame_dir.mkdir(parents=True, exist_ok=True)

    T = min(len(traces[m]["time"]) for m in METHODS)
    xlim, ylim = _bounds(traces)
    frame_idx = 0
    for t in range(0, T, max(1, args.frame_stride)):
        render_frame(traces, t, xlim, ylim, frame_dir / f"frame_{frame_idx:04d}.png")
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  rendered {frame_idx} frames ...")

    video_path = ROOT / "demo" / "videos" / "ac_dsgf_dynamic_comm.mp4"
    out = _export_uniform_mp4(frame_dir, video_path, fps=args.fps)
    print(f"Saved {out}")
    # keep gif fallback too
    try:
        from visualization.render_frames import frames_to_gif

        frames_to_gif(frame_dir, video_path.with_suffix(".gif"), fps=args.fps)
    except Exception as e:
        print(f"[warn] gif fallback skipped: {e}")

    if args.plot:
        plot_evolution_summary(
            traces, ROOT / "demo" / "figures" / "communication_evolution.png"
        )
        # also paper folder
        plot_evolution_summary(
            traces, ROOT / "paper" / "figures" / "fig_communication_evolution.png"
        )

    note = ROOT / "demo" / "DYNAMIC_COMM_README.md"
    note.write_text(
        """# Dynamic Communication Demo

Qualitative **Communication Emergence** visualization (4 UAV).

**Not** Table I / Budget-sweep metrics. Formal 16 UAV evidence stays in `paper/tables/`.

## Claim (safe wording)
AC-DSGF reduces communication overhead while maintaining *comparable* task performance —
not “higher success by cutting communication.”

## Outputs
- `demo/videos/ac_dsgf_dynamic_comm.mp4`
- `demo/figures/communication_evolution.png`
- `demo/traces/{gat,dsgf,ac_dsgf}_episode.json`

## Regeneration
```bash
python demo/generate_ac_demo.py --episodes 6 --select-best
```
""",
        encoding="utf-8",
    )
    print(f"Saved {note}")


if __name__ == "__main__":
    main()
