"""实验3 — 通信开销: Communication Cost。"""

from __future__ import annotations

import argparse

from experiments._eval import run_eval


def main():
    parser = argparse.ArgumentParser(description="Exp3: Communication cost")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    run_eval(args.config, label="communication")


if __name__ == "__main__":
    main()
