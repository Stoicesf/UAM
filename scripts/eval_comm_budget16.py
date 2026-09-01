"""Phase 1-2 — 16 UAV communication budget sweep (formal figure).

Requires AC-DSGF checkpoint: results/ac_dsgf/uav16/s{seed}/checkpoints/final.pt

Usage:
  python scripts/eval_comm_budget16.py --seed 42 --episodes 64 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out-dir", default="results/ac_dsgf/budget_sweep16")
    args, _ = parser.parse_known_args()

    ac_ckpt = ROOT / "results" / "ac_dsgf" / "uav16" / f"s{args.seed}" / "checkpoints" / "final.pt"
    if not ac_ckpt.exists():
        print(f"[blocked] missing {ac_ckpt}")
        print("Train first: python scripts/run_5seed_ac_dsgf16.py --seeds", args.seed)
        sys.exit(2)

    import scripts.eval_comm_budget as ecb

    ecb.METHODS = {
        "gat": {
            "exp": "configs/baseline16/gat_mappo.yaml",
            "ckpt": f"results/baseline16_seeds/gat/s{args.seed}/checkpoints/final.pt",
        },
        "dsgf": {
            "exp": "configs/baseline16/dsgf_v2.yaml",
            "ckpt": f"results/baseline16_seeds/dsgf/s{args.seed}/checkpoints/final.pt",
        },
        "ac_dsgf": {
            "exp": "configs/ac_dsgf/ac_dsgf_16uav.yaml",
            "ckpt": f"results/ac_dsgf/uav16/s{args.seed}/checkpoints/final.pt",
        },
    }

    # Rebuild argv for ecb.main()
    argv = [
        "eval_comm_budget",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--out-dir",
        args.out_dir,
    ]
    if args.plot:
        argv.append("--plot")
    sys.argv = argv

    # Redirect figure path after plot by monkeypatch
    _orig_plot = ecb.plot_budget

    def _plot16(rows, out):
        # always also write formal 16UAV figure name
        _orig_plot(rows, out)
        formal = ROOT / "paper" / "figures" / "fig_comm_budget16.png"
        if args.plot:
            _orig_plot(rows, formal)

    ecb.plot_budget = _plot16
    # Also write paper table as table_budget_sweep16.csv
    _orig_main = ecb.main

    def wrapped():
        _orig_main()
        src = ROOT / "paper" / "tables" / "table_budget_sweep.csv"
        dst = ROOT / "paper" / "tables" / "table_budget_sweep16.csv"
        if src.exists():
            dst.write_bytes(src.read_bytes())
            print(f"Copied → {dst}")

    wrapped()


if __name__ == "__main__":
    main()
