"""AC-DSGF++ training diagnostics — utility / gate / correlation CSV logs."""

from __future__ import annotations

import csv
from pathlib import Path


class ACPlusPlusRunLogger:
    """Writes Phase-5 smoke logs under <run_root>/logs/."""

    def __init__(self, run_root: str | Path):
        self.log_dir = Path(run_root) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "utility_mean": self.log_dir / "utility_mean.csv",
            "gate_mass": self.log_dir / "gate_mass.csv",
            "utility_corr": self.log_dir / "utility_corr.csv",
            "comm_precision": self.log_dir / "comm_precision.csv",
            "cud": self.log_dir / "cud.csv",
        }
        self._writers: dict[str, tuple] = {}

    def log_step(
        self,
        step: int,
        *,
        utility_mean: float,
        utility_std: float,
        gate_mass: float,
        utility_corr: float,
        utility_loss: float | None = None,
        comm_precision: float | None = None,
        cud: float | None = None,
    ) -> None:
        self._append(
            "utility_mean",
            {
                "step": step,
                "utility_mean": round(utility_mean, 6),
                "utility_std": round(utility_std, 6),
            },
        )
        self._append(
            "gate_mass",
            {"step": step, "gate_mass": round(gate_mass, 6)},
        )
        row = {
            "step": step,
            "corr_u_ustar": round(utility_corr, 6),
        }
        if utility_loss is not None:
            row["utility_loss"] = round(utility_loss, 6)
        if comm_precision is not None:
            row["comm_precision"] = round(comm_precision, 6)
        self._append("utility_corr", row)
        if comm_precision is not None:
            self._append(
                "comm_precision",
                {"step": step, "comm_precision": round(comm_precision, 6)},
            )
        if cud is not None:
            self._append(
                "cud",
                {"step": step, "cud": round(cud, 6)},
            )

    def _append(self, name: str, row: dict) -> None:
        path = self.paths[name]
        if name not in self._writers:
            f = open(path, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            self._writers[name] = (f, writer)
        else:
            f, writer = self._writers[name]
        writer.writerow(row)
        f.flush()

    def close(self) -> None:
        for f, _ in self._writers.values():
            f.close()
        self._writers.clear()
