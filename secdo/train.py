#!/usr/bin/env python
"""Unified SECDO training entry.

Usage:
  python -m secdo.train mode=pretrain --config secdo/configs/train/pretrain.yaml
  python -m secdo.train mode=joint --config secdo/configs/train/joint.yaml
  python -m secdo.train mode=online --config secdo/configs/train/online.yaml

Also accepts:
  python -m secdo.train --mode pretrain --config ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secdo.training.trainer import Trainer, TrainMode, load_train_config


def _parse_mode_token(argv: list[str]) -> tuple[TrainMode | None, list[str]]:
    """Support Hydra-like mode=pretrain tokens."""
    mode = None
    rest = []
    for a in argv:
        if a.startswith("mode="):
            mode = a.split("=", 1)[1]  # type: ignore[assignment]
        else:
            rest.append(a)
    return mode, rest  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    mode_tok, argv = _parse_mode_token(argv)
    p = argparse.ArgumentParser(description="SECDO unified trainer")
    p.add_argument("--mode", type=str, default=None, choices=["pretrain", "joint", "online"])
    p.add_argument("--config", type=str, default="secdo/configs/train/pretrain.yaml")
    args = p.parse_args(argv)
    mode: TrainMode = (args.mode or mode_tok or "pretrain")  # type: ignore[assignment]
    cfg = load_train_config(args.config, mode=mode)
    cfg.mode = mode
    Trainer(cfg).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
