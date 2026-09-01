"""实验1 — Baseline 对比: IPPO / MAPPO / DSGF，指标 Success Rate。"""

from __future__ import annotations

import argparse

from experiments._eval import run_eval


def main():
    parser = argparse.ArgumentParser(description="Exp1: Baseline comparison")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    run_eval(args.config, label="baseline")


if __name__ == "__main__":
    main()
