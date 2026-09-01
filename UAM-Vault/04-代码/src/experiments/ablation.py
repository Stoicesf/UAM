"""实验5 — 消融: 去掉 DSGF / Guide Reward / Sparse Attention。"""

from __future__ import annotations

import argparse

from experiments._eval import run_ablation


def main():
    parser = argparse.ArgumentParser(description="Exp5: Ablation study")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    run_ablation(args.config)


if __name__ == "__main__":
    main()
