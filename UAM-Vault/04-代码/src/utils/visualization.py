"""结果可视化 — 生成论文 figures/。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_learning_curve(steps: list, values: list, title: str, save_path: str):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, values)
    plt.xlabel("Steps")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
