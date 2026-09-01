"""实验4 — Scalability: 4 / 8 / 16 / 32 无人机。"""

from __future__ import annotations

import argparse

from experiments._eval import run_scalability


def main():
    parser = argparse.ArgumentParser(description="Exp4: Scalability")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    run_scalability(args.config)


if __name__ == "__main__":
    main()
