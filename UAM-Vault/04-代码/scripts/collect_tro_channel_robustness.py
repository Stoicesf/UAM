# -*- coding: utf-8 -*-
"""§6.5 Channel robustness (NO training, frozen checkpoints).

Factors (one at a time):
  packet loss p_l ∈ {0,0.1,0.3,0.5}
  delay τ_ms ∈ {0,20,50,100} → steps {0,1,2,4}
  bandwidth B/B_full ∈ {0.1,0.2,0.4,0.8}

Methods: ac_dsgf | dsgf (fixed sparse family) | full_attn

Claim: more stable coordination under degraded channels — not a robustness theorem.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import step_mdp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tro.collector.common import ROOT, as_float as _as_float, get_adapter, load_method, resolve_device

from algorithms.baseline.mappo import build_mappo
from algorithms.guided.mappo_guided import ACGuideAdapter, build_guided_mappo
from tro.channel import clear_policy_channel, install_channel_hooks, set_policy_channel
from tro.twin.message_compare import message_mean_l2
from utils.experiment import load_experiment_config
from utils.seed import set_seed
from utils.tro_run import create_run_dir, write_config

OUT = ROOT / "paper" / "ac_dsgf_tro" / "experiments" / "evidence_6_5_channel"
FIG_OUT = OUT / "figures"
FIG_PAPER = ROOT / "paper" / "ac_dsgf_tro" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_PAPER.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = (1234, 2026, 3407, 42, 8888)
LOSS = (0.0, 0.1, 0.3, 0.5)
DELAY_MS = (0, 20, 50, 100)
DELAY_STEPS = {0: 0, 20: 1, 50: 2, 100: 4}  # dt ≈ 25 ms
BANDWIDTH = (0.1, 0.2, 0.4, 0.8)

METHODS = {
    "ac_dsgf": (
        "AC-DSGF",
        "configs/ac_dsgf/ac_dsgf_16uav.yaml",
        "results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt",
    ),
    "dsgf": (
        "DSGF",
        "configs/baseline16/dsgf_v2.yaml",
        "results/baseline16_seeds/dsgf/s{seed}/checkpoints/final.pt",
    ),
    "full_attn": (
        "Full-Attention",
        "configs/baseline16/full_attention.yaml",
        "results/baseline16_seeds/transformer/s{seed}/checkpoints/final.pt",
    ),
}


def get_encoder(policy):
    ad = get_adapter(policy)
    return ad.encoder if ad is not None else None


def apply_bandwidth(policy, method: str, ratio: float | None) -> None:
    """AC-DSGF: hard budget_ratio. Others: no structural Π — leave as-is if None."""
    adapter = get_adapter(policy)
    if adapter is None:
        return
    if ratio is None or ratio >= 1.0:
        adapter.set_fixed_k(None)
        adapter.set_budget_ratio(None)
    else:
        adapter.set_fixed_k(None)
        adapter.set_budget_ratio(float(ratio))


@torch.no_grad()
def rollout(
    env,
    policy,
    *,
    method: str,
    episodes: int,
    max_steps: int,
    success_thr: float,
    packet_loss: float,
    delay_steps: int,
    bandwidth: float | None,
) -> dict:
    clear_policy_channel(policy)
    set_policy_channel(policy, packet_loss=packet_loss, delay_steps=delay_steps)
    apply_bandwidth(policy, method, bandwidth)

    adapter = get_adapter(policy)
    encoder = get_encoder(policy)
    rewards, successes, costs, edges, epsilons = [], [], [], [], []

    for _ in range(episodes):
        # reset delay buffers between episodes
        clear_policy_channel(policy)
        set_policy_channel(policy, packet_loss=packet_loss, delay_steps=delay_steps)
        apply_bandwidth(policy, method, bandwidth)

        td = env.reset()
        ep_r, ep_s, ep_c, ep_e, ep_eps, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_eps = 0
        for _t in range(max_steps):
            td = policy(td)
            # Channel stress can zero all edges → NaN actions; sanitize for VMAS.
            act = td.get(("agents", "action"))
            if act is not None and torch.is_tensor(act) and bool(act.isnan().any()):
                td[("agents", "action")] = torch.nan_to_num(act, nan=0.0)
            loc = td.get(("agents", "loc"))
            if loc is not None and torch.is_tensor(loc) and bool(loc.isnan().any()):
                td[("agents", "loc")] = torch.nan_to_num(loc, nan=0.0)
            if adapter is not None and adapter.last_topology_diag is not None:
                diag = adapter.last_topology_diag
                if "C_t" in diag:
                    ep_c += _as_float(diag["C_t"])
                A = diag.get("A_t")
                if A is not None and torch.is_tensor(A):
                    ep_e += float((A > 0).float().sum().item())
                n += 1
                # twin ε_G optional (expensive); enable with --with-epsilon
                if (
                    getattr(rollout, "with_epsilon", False)
                    and encoder is not None
                    and hasattr(encoder, "twin_forward")
                ):
                    obs = td.get(("agents", "observation"))
                    obs_b = obs.unsqueeze(0) if obs.dim() == 2 else obs
                    try:
                        twin = encoder.twin_forward(obs_b, sparse_mode="fixed_k", k_fixed=4)
                        ep_eps += message_mean_l2(
                            twin["full"]["message"], twin["sparse"]["message"]
                        )
                        n_eps += 1
                    except Exception:
                        pass
            elif method == "full_attn":
                ep_c += 16 * 15 * (1.0 - packet_loss)
                ep_e += 16 * 15 * (1.0 - packet_loss)
                n += 1
            else:
                # DSGF / other non-AC: radius-neighborhood edge count as C proxy
                from utils.comm_metrics import compute_sparse_communication_stats

                obs = td.get(("agents", "observation"))
                if obs is not None:
                    obs_b = obs.unsqueeze(0) if obs.dim() == 2 else obs
                    stats = compute_sparse_communication_stats(obs_b, 0.5)
                    # approximate delivered edges under i.i.d. loss
                    ep_c += float(stats["communication_cost"]) * (1.0 - packet_loss)
                    ep_e += float(stats["communication_cost"]) * (1.0 - packet_loss)
                    n += 1

            td = env.step(td)
            obs_n = td.get(("next", "agents", "observation"))
            if obs_n is not None:
                goal = obs_n[0, :, 4:6] if obs_n.dim() == 3 else obs_n[:, 4:6]
                ep_s = float((goal.norm(dim=-1) < success_thr).float().mean().item())
            rew = td.get(("next", "agents", "reward"))
            if rew is not None:
                ep_r += float(rew.mean().item())
            done = td.get(("next", "done"))
            if done is not None and bool(done.any()):
                break
            td = step_mdp(td)

        rewards.append(ep_r)
        successes.append(ep_s)
        costs.append(ep_c / max(n, 1))
        edges.append(ep_e / max(n, 1))
        if n_eps:
            epsilons.append(ep_eps / n_eps)

    C = float(np.mean(costs))
    return {
        "reward": float(np.mean(rewards)),
        "success": float(np.mean(successes)),
        "C": C,
        "C_eff": C * (1.0 - packet_loss),
        "E_mean": float(np.mean(edges)),
        "epsilon_G": float(np.mean(epsilons)) if epsilons else float("nan"),
    }


FIELDS = [
    "factor",
    "method",
    "display",
    "seed",
    "p_loss",
    "delay_ms",
    "delay_steps",
    "bandwidth",
    "reward",
    "success",
    "C",
    "C_eff",
    "E_mean",
    "epsilon_G",
]


def _write_csv(rows: list[dict]) -> None:
    path = OUT / "channel_statistics.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def aggregate(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    g: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["factor"], r["method"], r["p_loss"], r["delay_ms"], r["bandwidth"])
        g[key].append(r)
    out = []
    for (factor, method, pl, dm, bw), rs in sorted(g.items(), key=lambda x: str(x[0])):
        def mean(k):
            vals = [float(x[k]) for x in rs if x.get(k) not in ("", None) and np.isfinite(float(x[k]))]
            return float(np.mean(vals)) if vals else float("nan")

        def std(k):
            vals = [float(x[k]) for x in rs if x.get(k) not in ("", None) and np.isfinite(float(x[k]))]
            return float(np.std(vals)) if vals else float("nan")

        out.append(
            {
                "factor": factor,
                "method": method,
                "display": rs[0]["display"],
                "p_loss": pl,
                "delay_ms": dm,
                "bandwidth": bw,
                "n_seeds": len(rs),
                "reward_mean": mean("reward"),
                "reward_std": std("reward"),
                "success_mean": mean("success"),
                "C_mean": mean("C"),
                "C_eff_mean": mean("C_eff"),
                "E_mean": mean("E_mean"),
                "epsilon_G_mean": mean("epsilon_G"),
            }
        )
    return out


def plot_fig6(agg: list[dict]) -> Path | None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    colors = {"ac_dsgf": "#C44E52", "dsgf": "#4C72B0", "full_attn": "#55A868"}
    labels = {"ac_dsgf": "AC-DSGF", "dsgf": "DSGF", "full_attn": "Full-Attn"}

    # (a) loss
    ax = axes[0]
    for m, c in colors.items():
        pts = sorted(
            [r for r in agg if r["factor"] == "loss" and r["method"] == m],
            key=lambda x: float(x["p_loss"]),
        )
        if not pts:
            continue
        ax.errorbar(
            [100 * float(p["p_loss"]) for p in pts],
            [p["reward_mean"] for p in pts],
            yerr=[p["reward_std"] for p in pts],
            marker="o",
            color=c,
            label=labels[m],
            capsize=3,
        )
    ax.set_xlabel(r"Packet loss $p_\ell$ (%)")
    ax.set_ylabel(r"Return $J$")
    ax.set_title("(a) Packet loss")
    ax.grid(True, alpha=0.3)

    # (b) delay
    ax = axes[1]
    for m, c in colors.items():
        pts = sorted(
            [r for r in agg if r["factor"] == "delay" and r["method"] == m],
            key=lambda x: float(x["delay_ms"]),
        )
        if not pts:
            continue
        ax.errorbar(
            [float(p["delay_ms"]) for p in pts],
            [p["reward_mean"] for p in pts],
            yerr=[p["reward_std"] for p in pts],
            marker="o",
            color=c,
            label=labels[m],
            capsize=3,
        )
    ax.set_xlabel(r"Delay $\tau$ (ms)")
    ax.set_ylabel(r"Return $J$")
    ax.set_title("(b) Delay")
    ax.grid(True, alpha=0.3)

    # (c) bandwidth
    ax = axes[2]
    for m, c in colors.items():
        pts = sorted(
            [r for r in agg if r["factor"] == "bandwidth" and r["method"] == m],
            key=lambda x: float(x["bandwidth"]),
        )
        if not pts:
            continue
        ax.errorbar(
            [float(p["bandwidth"]) for p in pts],
            [p["reward_mean"] for p in pts],
            yerr=[p["reward_std"] for p in pts],
            marker="o",
            color=c,
            label=labels[m],
            capsize=3,
        )
    ax.set_xlabel(r"$B / B_{\mathrm{full}}$")
    ax.set_ylabel(r"Return $J$")
    ax.set_title("(c) Bandwidth")
    ax.grid(True, alpha=0.3)

    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_PAPER / "Fig6_channel_robustness.png"
    fig.savefig(path, dpi=200)
    fig.savefig(FIG_OUT / "Fig6_channel_robustness.png", dpi=200)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser(description="T-RO §6.5 channel robustness")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        choices=list(METHODS.keys()),
    )
    ap.add_argument(
        "--factors",
        nargs="+",
        default=["loss", "delay", "bandwidth"],
        choices=["loss", "delay", "bandwidth"],
    )
    ap.add_argument(
        "--with-epsilon",
        action="store_true",
        help="also log twin epsilon_G (slow)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip cells already present in channel_statistics.csv",
    )
    args = ap.parse_args()

    device = resolve_device(args.device)
    install_channel_hooks()
    rollout.with_epsilon = bool(args.with_epsilon)  # type: ignore[attr-defined]
    run_dir = create_run_dir(prefix="tro_evidence_6_5_channel")
    write_config(
        run_dir,
        {
            "section": "6.5",
            "seeds": args.seeds,
            "episodes": args.episodes,
            "methods": args.methods,
            "factors": args.factors,
            "delay_step_map": DELAY_STEPS,
            "training": False,
            "resume": bool(args.resume),
        },
    )
    print(f"device={device} | run={run_dir}", flush=True)

    rows: list[dict] = []
    done: set[tuple] = set()
    csv_path = OUT / "channel_statistics.csv"
    if args.resume and csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
                done.add(
                    (
                        r["method"],
                        str(r["seed"]),
                        r["factor"],
                        str(r.get("p_loss", "")),
                        str(r.get("delay_ms", "")),
                        str(r.get("bandwidth", "")),
                    )
                )
        print(f"resume: loaded {len(rows)} rows, {len(done)} done keys", flush=True)

    def _key(method, seed, factor, p_loss, delay_ms, bandwidth) -> tuple:
        return (
            method,
            str(seed),
            factor,
            str(p_loss),
            str(delay_ms),
            str(bandwidth if bandwidth is not None else ""),
        )
    for method in args.methods:
        for seed in args.seeds:
            set_seed(seed)
            print(f"[{method}] seed={seed} loading…", flush=True)
            cfg, env, policy, display = load_method(METHODS, method, seed, device)
            thr = float(cfg.get("env", {}).get("success_threshold", 0.3))
            max_steps = int(cfg.get("env", {}).get("max_steps", 128) or 128)

            if "loss" in args.factors:
                for p in LOSS:
                    if _key(method, seed, "loss", p, 0, "") in done:
                        continue
                    r = rollout(
                        env,
                        policy,
                        method=method,
                        episodes=args.episodes,
                        max_steps=max_steps,
                        success_thr=thr,
                        packet_loss=p,
                        delay_steps=0,
                        bandwidth=None,
                    )
                    row = {
                        "factor": "loss",
                        "method": method,
                        "display": display,
                        "seed": seed,
                        "p_loss": p,
                        "delay_ms": 0,
                        "delay_steps": 0,
                        "bandwidth": "",
                        **r,
                    }
                    rows.append(row)
                    done.add(_key(method, seed, "loss", p, 0, ""))
                    print(
                        f"  loss p={p:.1f} J={r['reward']:.3f} S={r['success']:.3f} "
                        f"C={r['C']:.2f} Ceff={r['C_eff']:.2f} E={r['E_mean']:.1f}",
                        flush=True,
                    )
                    _write_csv(rows)

            if "delay" in args.factors:
                for ms in DELAY_MS:
                    ds = DELAY_STEPS[ms]
                    if _key(method, seed, "delay", 0.0, ms, "") in done:
                        continue
                    r = rollout(
                        env,
                        policy,
                        method=method,
                        episodes=args.episodes,
                        max_steps=max_steps,
                        success_thr=thr,
                        packet_loss=0.0,
                        delay_steps=ds,
                        bandwidth=None,
                    )
                    row = {
                        "factor": "delay",
                        "method": method,
                        "display": display,
                        "seed": seed,
                        "p_loss": 0.0,
                        "delay_ms": ms,
                        "delay_steps": ds,
                        "bandwidth": "",
                        **r,
                    }
                    rows.append(row)
                    done.add(_key(method, seed, "delay", 0.0, ms, ""))
                    print(
                        f"  delay {ms}ms ({ds} steps) J={r['reward']:.3f} S={r['success']:.3f}",
                        flush=True,
                    )
                    _write_csv(rows)

            if "bandwidth" in args.factors:
                for bw in BANDWIDTH:
                    # Bandwidth primarily meaningful for AC-DSGF projection;
                    # still evaluate others at nominal (no Π) once for reference at bw=1 skip.
                    if method != "ac_dsgf":
                        # For non-AC methods, only evaluate at full (skip grid) once per seed
                        # under factor bandwidth — record as N/A structural match.
                        # Still run AC-style ratio only for ac_dsgf.
                        continue
                    if _key(method, seed, "bandwidth", 0.0, 0, bw) in done:
                        continue
                    r = rollout(
                        env,
                        policy,
                        method=method,
                        episodes=args.episodes,
                        max_steps=max_steps,
                        success_thr=thr,
                        packet_loss=0.0,
                        delay_steps=0,
                        bandwidth=bw,
                    )
                    row = {
                        "factor": "bandwidth",
                        "method": method,
                        "display": display,
                        "seed": seed,
                        "p_loss": 0.0,
                        "delay_ms": 0,
                        "delay_steps": 0,
                        "bandwidth": bw,
                        **r,
                    }
                    rows.append(row)
                    done.add(_key(method, seed, "bandwidth", 0.0, 0, bw))
                    print(
                        f"  bw={bw} J={r['reward']:.3f} S={r['success']:.3f} C={r['C']:.2f}",
                        flush=True,
                    )
                    _write_csv(rows)

            del env, policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # For bandwidth panel: also add DSGF/Full-Attn nominal point as reference at bw=1.0
    # (optional — skip to keep protocol clean)

    agg = aggregate(rows)
    fig = plot_fig6(agg)
    report = {
        "n_rows": len(rows),
        "aggregate": agg,
        "fig6": str(fig) if fig else None,
        "claim": (
            "AC-DSGF maintains more stable coordination performance under "
            "degraded communication conditions."
        ),
        "run_dir": str(run_dir),
    }
    (OUT / "evidence_6_5_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Wrote", OUT / "channel_statistics.csv", flush=True)
    print("Wrote", OUT / "evidence_6_5_report.json", flush=True)
    if fig:
        print("Wrote", fig, flush=True)


if __name__ == "__main__":
    main()
