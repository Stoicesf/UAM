"""日志 — TensorBoard + 控制台。"""

from __future__ import annotations

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
