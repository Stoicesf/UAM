"""Stage II entry — thin wrapper around Trainer(mode=joint)."""

from __future__ import annotations

import argparse

from secdo.training.trainer import Trainer, load_train_config


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="secdo/configs/train/joint.yaml")
    args = p.parse_args(argv)
    cfg = load_train_config(args.config, mode="joint")
    Trainer(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
