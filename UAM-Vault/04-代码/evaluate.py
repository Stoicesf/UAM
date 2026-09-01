"""评估入口 — 加载 checkpoint 运行第五章实验。"""

from __future__ import annotations

import argparse

import yaml

from experiments._eval import run_ablation, run_eval, run_scalability
from utils.seed import set_seed

HANDLERS = {
    "baseline": lambda c: run_eval(c, label="baseline"),
    "dsfg": lambda c: run_eval(c, label="dsfg"),
    "ablation": run_ablation,
    "scalability": run_scalability,
    "communication": lambda c: run_eval(c, label="communication"),
}


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--experiment", default=None, choices=list(HANDLERS))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    exp = args.experiment or cfg.get("experiment", "baseline")
    HANDLERS[exp](args.config)


if __name__ == "__main__":
    main()
