"""Experiment run management — config, seed, metrics, checkpoints."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentMeta:
    experiment_id: str
    stage: int
    name: str
    run_name: str
    seed: int
    started_at: str
    config_path: str
    guidance_mode: str = "none"
    extra: dict = field(default_factory=dict)


CATEGORY_MAP = {
    "none": "baseline",
    "mlp": "guide",
    "sparse": "sparse",
    "graph": "graph",
    "gat": "graph",
    "full": "dsgf",
    "dsfg": "dsgf",
    "ac_dsgf": "ac_dsgf",
    "ac-dsgf": "ac_dsgf",
    "ac_dsgf_pp": "ac_dsgf_pp",
    "ac-dsgf-pp": "ac_dsgf_pp",
    "ac_dsgf++": "ac_dsgf_pp",
}


class ExperimentRun:
    """One experiment run with reproducible artifacts."""

    def __init__(
        self,
        exp_cfg: dict[str, Any],
        run_name: str | None = None,
        config_path: str | None = None,
        output_root: str | Path | None = None,
    ):
        meta = exp_cfg.get("experiment", {})
        self.experiment_id = meta.get("id", "unknown")
        self.stage = meta.get("stage", 0)
        self.name = meta.get("name", self.experiment_id)
        self.seed = exp_cfg.get("train", {}).get("seed", 42)
        self.guidance_mode = exp_cfg.get("guidance", {}).get("mode", "none")
        output = exp_cfg.get("output", {})
        self.category = output.get("category") or CATEGORY_MAP.get(self.guidance_mode, "other")
        self.tag = output.get("tag")
        self.comm_sweep = exp_cfg.get("comm_sweep")

        ts = datetime.now().strftime("%m%d_%H%M")
        self.run_name = run_name or f"{self.experiment_id}_{ts}"
        base = Path(output.get("results_dir", "results"))
        self.root = Path(output_root) if output_root else base / self.category / self.run_name
        self.log_dir = self.root / "tensorboard"
        self.ckpt_dir = self.root / "checkpoints"
        self.plots_dir = self.root / "plots"
        self.figures_dir = Path(output.get("figures_dir", "figures"))
        self.metrics_path = self.root / "metrics.csv"
        self.action_alignment_path = self.root / "action_alignment.csv"
        self.communication_path = self.root / "communication.csv"
        self.summary_path = self.root / "summary.json"

        for d in (self.root, self.log_dir, self.ckpt_dir, self.plots_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

        if config_path:
            shutil.copy2(config_path, self.root / "config.yaml")
        with open(self.root / "config_resolved.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(exp_cfg, f, allow_unicode=True, sort_keys=False)

        self.meta = ExperimentMeta(
            experiment_id=self.experiment_id,
            stage=self.stage,
            name=self.name,
            run_name=self.run_name,
            seed=self.seed,
            started_at=datetime.now().isoformat(timespec="seconds"),
            config_path=str(config_path or ""),
            guidance_mode=self.guidance_mode,
        )
        with open(self.root / "meta.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.meta), f, indent=2, ensure_ascii=False)

        self._csv_writers: dict[str, tuple[Any, csv.DictWriter, list[str]]] = {}
        self._open_csv("metrics", self.metrics_path)
        self._open_csv("action_alignment", self.action_alignment_path)
        self._open_csv("communication", self.communication_path)

    def _open_csv(self, name: str, path: Path):
        f = open(path, "w", newline="", encoding="utf-8")
        self._csv_writers[name] = (f, None, [])

    def _write_csv_row(self, name: str, row: dict):
        f, writer, fields = self._csv_writers[name]
        if writer is None:
            fields = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            self._csv_writers[name] = (f, writer, fields)
        writer.writerow(row)
        f.flush()

    def log_metrics(self, step: int, metrics: dict[str, float]):
        self._write_csv_row("metrics", {"step": step, **metrics})

    def log_action_alignment(self, step: int, alignment: float, reward_guide: float):
        self._write_csv_row(
            "action_alignment",
            {"step": step, "action_alignment": alignment, "reward_guide": reward_guide},
        )

    def log_communication(
        self,
        step: int,
        comm_cost: float,
        comm_ratio: float = 0.0,
        sparse_edges: float | None = None,
        full_graph_edges: float | None = None,
    ):
        row = {
            "step": step,
            "communication_cost": comm_cost,
            "communication_ratio": comm_ratio,
        }
        if sparse_edges is not None:
            row["sparse_edges"] = sparse_edges
        if full_graph_edges is not None:
            row["full_graph_edges"] = full_graph_edges
        self._write_csv_row("communication", row)

    def save_training_curves(self, metrics_history: list[dict], reward_curve: list[float]):
        """Save numpy curves for paper figures (success, collision, reward)."""
        try:
            import numpy as np
        except ImportError:
            return

        curves = {
            "reward_curve": np.array(reward_curve, dtype=np.float32),
            "success_curve": np.array(
                [m.get("success_rate", 0.0) for m in metrics_history], dtype=np.float32
            ),
            "collision_curve": np.array(
                [m.get("collision_rate", 0.0) for m in metrics_history], dtype=np.float32
            ),
            "path_length_curve": np.array(
                [m.get("path_length", 0.0) for m in metrics_history], dtype=np.float32
            ),
        }
        if metrics_history and "sparse_edges" in metrics_history[0]:
            curves["communication_curve"] = np.array(
                [m.get("sparse_edges", 0.0) for m in metrics_history], dtype=np.float32
            )
        for name, arr in curves.items():
            np.save(self.root / f"{name}.npy", arr)

    def finalize(
        self,
        reward_curve: list[float],
        metrics_history: list[dict] | None = None,
        eval_stats: dict | None = None,
        extra: dict | None = None,
    ):
        from utils.training_helpers import build_paper_summary

        paper = build_paper_summary(reward_curve, metrics_history or [], eval_stats)
        summary = {
            **asdict(self.meta),
            "category": self.category,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "num_iterations": len(reward_curve),
            "final_reward": reward_curve[-1] if reward_curve else None,
            "max_reward": max(reward_curve) if reward_curve else None,
            "reward_curve": reward_curve,
            "paper_metrics": paper,
            **paper,
        }
        if extra:
            summary.update(extra)
        if self.comm_sweep:
            summary["comm_sweep"] = self.comm_sweep
        if metrics_history:
            comm_vals = [m["sparse_edges"] for m in metrics_history if "sparse_edges" in m]
            if comm_vals:
                summary["communication_cost_mean"] = round(sum(comm_vals) / len(comm_vals), 4)
        self.save_training_curves(metrics_history or [], reward_curve)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if self.tag:
            with open(self.root / "TAG.txt", "w", encoding="utf-8") as f:
                f.write(f"{self.tag}\n")

        for f, _, _ in self._csv_writers.values():
            f.close()

    @property
    def checkpoint_latest(self) -> Path:
        return self.ckpt_dir / "latest.pt"

    @property
    def checkpoint_final(self) -> Path:
        return self.ckpt_dir / "final.pt"


def load_experiment_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    defaults_train = Path("configs/train.yaml")
    defaults_env = Path("configs/environment.yaml")
    if defaults_train.exists():
        with open(defaults_train, encoding="utf-8") as f:
            base_train = yaml.safe_load(f) or {}
        cfg["train"] = {**base_train, **cfg.get("train", {})}
    if defaults_env.exists():
        with open(defaults_env, encoding="utf-8") as f:
            base_env = yaml.safe_load(f) or {}
        cfg["env"] = {**base_env, **cfg.get("env", {})}
    return cfg


def normalize_train_frames(train_cfg: dict, smoke: bool = False, gate: int | None = None):
    if smoke:
        train_cfg["total_frames"] = 2048
        train_cfg["frames_per_batch"] = 2048
        train_cfg["save_interval"] = 2048
    elif gate == 2:
        train_cfg["total_frames"] = 10_240  # ~10k (5 x 2048)
        train_cfg["save_interval"] = 10_240
        if not train_cfg.get("lock_eval_episodes"):
            train_cfg["eval_episodes"] = 32
    elif gate == 3:
        if not train_cfg.get("lock_total_frames"):
            train_cfg["total_frames"] = 102_400
        train_cfg["save_interval"] = train_cfg.get("save_interval", 10_240)
        train_cfg["eval_episodes"] = train_cfg.get("eval_episodes", 200)
    else:
        fpb = train_cfg.get("frames_per_batch", 2048)
        tf = train_cfg.get("total_frames", 102_400)
        if tf % fpb != 0:
            train_cfg["total_frames"] = (tf // fpb) * fpb
