"""Freeze DSGF v2 reference artifacts for paper reproducibility."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "dsgf" / "dsfg_v2_102k"
DST = ROOT / "results" / "frozen_dsgf" / "dsgf_v2_final"
CFG = ROOT / "configs" / "dsgf" / "dsgf_v2_frozen.yaml"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CFG, DST / "config.yaml")
    if SRC.exists():
        for name in ("summary.json", "metrics.csv", "config_resolved.yaml", "meta.json"):
            src = SRC / name
            if src.exists():
                shutil.copy2(src, DST / name)
        ckpt_src = SRC / "checkpoints"
        if ckpt_src.exists():
            shutil.copytree(ckpt_src, DST / "checkpoints", dirs_exist_ok=True)
    meta = {
        "frozen": True,
        "algorithm": "DSGF v2",
        "residual_policy": True,
        "use_guidance_reward": False,
        "frames": 102400,
        "seed": 42,
        "note": "Do not modify residual structure, beta decay, or policy input after freeze.",
    }
    if (SRC / "summary.json").exists():
        with open(SRC / "summary.json", encoding="utf-8") as f:
            s = json.load(f)
        meta["paper_metrics"] = s.get("paper_metrics", {})
    with open(DST / "FREEZE.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Frozen DSGF v2 -> {DST}")


if __name__ == "__main__":
    main()
