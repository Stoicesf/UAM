"""Watch 5-seed batch; auto-run P0 pipeline when all 20 summaries exist.

Usage:
  python scripts/watch_and_run_pipeline.py
  python scripts/watch_and_run_pipeline.py --poll-secs 300
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "pipeline_watch.log"
TARGET = 20


def _load_count_fn():
    spec = importlib.util.spec_from_file_location(
        "pipeline_after_5seed",
        ROOT / "scripts" / "pipeline_after_5seed.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.count_summaries


def log(msg: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-secs", type=int, default=300)
    args = parser.parse_args()

    count_summaries = _load_count_fn()
    log(f"Watching for {TARGET} baseline16_seeds summaries (poll={args.poll_secs}s)")

    while True:
        n, missing = count_summaries()
        log(f"Progress: {n}/{TARGET}")
        if n >= TARGET:
            log("All summaries ready — starting pipeline_after_5seed.py")
            rc = subprocess.call(
                [sys.executable, str(ROOT / "scripts" / "pipeline_after_5seed.py")],
                cwd=str(ROOT),
            )
            log(f"Pipeline exit code: {rc}")
            sys.exit(rc)

        time.sleep(args.poll_secs)


if __name__ == "__main__":
    main()
