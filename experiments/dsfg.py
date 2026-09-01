"""DSGF 完整方法评估。"""

from __future__ import annotations

import argparse

from experiments._eval import run_eval


def main():
    parser = argparse.ArgumentParser(description="DSGF-HRL evaluation")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    run_eval(args.config, label="dsfg")


if __name__ == "__main__":
    main()
