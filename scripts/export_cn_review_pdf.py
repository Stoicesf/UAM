# -*- coding: utf-8 -*-
"""Build a simple multi-page PDF review pack (figures + key tables)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1] / "paper" / "ac_dsgf_cn"
FIG = ROOT / "figures"
OUT = ROOT / "AC_DSGF_CN_PDF.pdf"


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.08, 0.06, 0.84, 0.88])
    ax.axis("off")
    ax.text(0, 0.98, title, fontsize=16, fontweight="bold", va="top", transform=ax.transAxes)
    y = 0.90
    for line in lines:
        ax.text(0, y, line, fontsize=10, va="top", transform=ax.transAxes, family="DejaVu Sans")
        y -= 0.035
        if y < 0.05:
            break
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, title, path: Path):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.97)
    ax = fig.add_axes([0.08, 0.12, 0.84, 0.80])
    ax.axis("off")
    if path.exists():
        img = plt.imread(str(path))
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"missing {path.name}", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    with PdfPages(str(OUT)) as pdf:
        text_page(
            pdf,
            "AC-DSGF Chinese Review Pack (Figures)",
            [
                "Full text: AC_DSGF_CN.md / AC_DSGF_CN_advisor.docx",
                "Claim: comparable Success + Comm down ~2 orders of magnitude",
                "NOT: Success leadership / causal utility / AC-DSGF++",
                "",
                "Table I (N=16, 5 seeds, 200-ep):",
                "  DSGF     Success 4.01+/-1.68  Comm 39.27  CEI 0.001",
                "  AC-DSGF  Success 3.95+/-0.83  Comm 0.43   CEI 0.091",
                "",
                "Runtime N=16 CPU: GAT 1.66 / DSGF 3.35 / AC 3.94 ms",
                "",
                "Future: ROS2/Gazebo · Hardware · Comm-Motion Co-design",
            ],
        )
        for name, title in [
            ("Fig1_framework.png", "Fig.1 Framework"),
            ("Fig2_motivation.png", "Fig.2 Motivation"),
            ("Fig3_algorithm_flow.png", "Fig.3 Algorithm Pipeline"),
            ("Fig4_pareto.png", "Fig.4 Pareto (core)"),
            ("Fig5_behavior.png", "Fig.5 Behavior"),
            ("Fig6_runtime.png", "Fig.6 Runtime"),
        ]:
            image_page(pdf, title, FIG / name)
    print("saved", OUT)


if __name__ == "__main__":
    main()
