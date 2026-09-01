# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(r"f:\UAM")
EN = ROOT / "paper" / "ac_dsgf" / "AC_DSGF_EN.md"
CN = ROOT / "paper" / "ac_dsgf_cn" / "AC_DSGF_CN.md"
MOCK_PATH = ROOT / "paper" / "ac_dsgf" / "response_mock.md"

EN_INSERT = r"""**On AC-random ≥ AC-full Success.** Random sparsification may occasionally improve task performance due to noise reduction, but it does **not** provide communication-efficiency guarantees under equivalent budgets.

> **Although AC-random achieves slightly higher Success than AC-full, its SCAM is ~2,600× larger (20.32 vs 0.0077), violating the communication budget by orders of magnitude. Under a hard budget constraint (e.g., ≤\(K\) edges per agent), AC-random cannot guarantee feasible scheduling, whereas AC-full’s sparse topology is directly deployable.**

Opening all edges (\(g{=}A\)) restores dense intensity without reproducing the high-CEI regime of AC-full. Gains come from **jointly learned selective topology**, not arbitrary random sparsification.

The advantage of AC-DSGF is not maximizing Success at any cost, but achieving **comparable or slightly lower Success with near-zero activation density**, which is the prerequisite for scaling to bandwidth-limited UAV swarms.

"""

CN_INSERT = r"""**关于 AC-random ≥ AC-full Success。** 随机稀疏化可能因噪声抑制而偶然提升任务表现，但**不提供等价预算下的通信效率保证**。

> **尽管 AC-random 的 Success 略高于 AC-full，其 SCAM 约为后者的 2,600 倍（20.32 vs 0.0077），在数量级上违反通信预算。在硬预算约束下（例如每智能体 ≤\(K\) 条边），AC-random 无法保证可行调度；而 AC-full 的稀疏拓扑可直接部署。**

全开边（\(g{=}A\)）恢复稠密强度，但无法复现 AC-full 的高 CEI 工作区。收益来自**联合学习的选择性拓扑**，而非任意随机稀疏化。

AC-DSGF 的优势不在于不计代价地最大化 Success，而在于以**近零激活密度取得可比或略低的 Success**——这是扩展到带宽受限无人机蜂群的前提。

"""

EN_INDEX = r"""## Figure Index

| Fig | File | Content |
|-----|------|---------|
| 0 | `figures/Fig0_motivation_topology.png` | Predefined → task-driven topology |
| 1 | `../ac_dsgf_cn/figures/Fig1_framework.png` | Framework |
| 2 | `../ac_dsgf_cn/figures/Fig3_algorithm_flow.png` | Pipeline |
| 3 | `../ac_dsgf_cn/figures/Fig4_pareto.png` | Success—SCAM |
| 4 | `../ac_dsgf_cn/figures/Fig7_comm_density.png` | Budget saturation |
| 5 | `../ac_dsgf_cn/figures/Fig5_behavior.png` | Behavior |
| 6 | `figures/Fig9_comm_density_scaling.png` | Density scaling \(\eta_N\) |
| 7 | `figures/Fig10_gate_distribution.png` | Selectivity |
| 8 | `figures/Fig11_topk_deploy.png` | Top-\(K\) deployment |
| 9 | `figures/Fig12_edge_importance.png` | Edge importance |
| 10 | `figures/Fig13_comm_pareto.png` | Budget–performance operating curve |
| 11 | `figures/Fig14_hard_topk.png` | Fixed hard Top-\(K\) |

Filenames retain historical prefixes; **display numbers are consecutive 0–11** and match in-text citations.
"""

CN_INDEX = r"""## 附图清单

| Fig | 文件 | 内容 |
|-----|------|------|
| 0 | `../ac_dsgf/figures/Fig0_motivation_topology.png` | 预设 → 任务驱动拓扑 |
| 1 | `figures/Fig1_framework.png` | 框架 |
| 2 | `figures/Fig3_algorithm_flow.png` | 流水线 |
| 3 | `figures/Fig4_pareto.png` | Success—SCAM |
| 4 | `figures/Fig7_comm_density.png` | 预算饱和 |
| 5 | `figures/Fig5_behavior.png` | 行为 |
| 6 | `../ac_dsgf/figures/Fig9_comm_density_scaling.png` | 密度标度 |
| 7 | `../ac_dsgf/figures/Fig10_gate_distribution.png` | 选择性 |
| 8 | `../ac_dsgf/figures/Fig11_topk_deploy.png` | Top-\(K\) 部署 |
| 9 | `../ac_dsgf/figures/Fig12_edge_importance.png` | 边重要性 |
| 10 | `../ac_dsgf/figures/Fig13_comm_pareto.png` | 预算—性能工作曲线 |
| 11 | `../ac_dsgf/figures/Fig14_hard_topk.png` | 固定硬 Top-\(K\) |

文件名保留历史前缀；**正文显示图号连续为 0–11**，与引用一致。
"""

MOCK_TEXT = r'''# Response Mock — AC-DSGF v1（RA-L 投稿防御）

**定位一句：** We treat communication topology as a first-class decision variable under soft budgets, learning *when and with whom* to communicate. Our advantage is **SCAM**—task-effective coordination at near-zero activation density—not brute-force Success.

---

## Q1 — Why not train with hard Top-K from the start?

**Attack:** Soft gates are unnecessary; just optimize discrete Top-K.

**Response:**
1. Soft gates \(g_{ij}\in[0,1]\) enable differentiable exploration under \(\mathbb{E}[R-\lambda_c C]\).
2. Hard Top-K is a **deployment approximation** of the learned soft pattern (Fig. 8), not a replacement training objective.
3. Under matched hard \(K\), AC is **comparable** to Distance/Random (Table 2b / Fig. 11); the primary gain appears in the **soft SCAM regime** (Table 1), where activation is orders of magnitude lower than dense baselines.
4. We do **not** claim Top-K training would be inferior; we claim soft training + hard deployment is a practical pipeline.

---

## Q2 — Why is AC-random Success higher than AC-full (Table 2)?

**Attack:** If random gates get 28.5% vs AC-full 24.6%, why learn?

**Response:**
1. **Magnitude gap:** AC-random SCAM is **~2,600×** larger (20.32 vs 0.0077). Higher Success without a budget is not the objective.
2. Under a **hard budget** (≤\(K\) edges/agent), random selection is not a feasible scheduler guarantee; AC-full’s sparse topology is deployable as-is.
3. Edge-importance ablation (Fig. 9): dropping top-\(g\) edges hurts tasks; dropping bottom/random does not → learned ranking is task-aligned, not chance.
4. Thesis sentence: advantage = **comparable Success at near-zero activation density**, the prerequisite for bandwidth-limited swarms.

---

## Q3 — Prop.1 is only an upper bound; is it tight?

**Attack:** \(\eta_N\le K/(N-1)\) is trivial degree counting.

**Response:**
1. Prop.1 is intentionally a **degree-constraint scaling interpretation**, not a learning-convergence theorem and not a claim of equality \(|E_t|=KN\).
2. Empirically, AC maintains low normalized density as \(N\) grows (Fig. 6), consistent with \(\mathcal{O}(1/N)\) decay versus denser baselines.
3. Tightness would require characterizing realized degree vs \(K\); we report measured SCAM/\(\eta_N\) rather than claiming a tight analytic constant.
4. Prop.2/3 remain interpretive (action stability; induced budgeted selection)—aligned with RA-L evidence standards.

---

## Quick extras (if raised)

| Attack | One-liner |
|--------|-----------|
| Soft Mass ≠ packets | SCAM is a training proxy; deployment uses threshold / Top-K hard edges. |
| Attention = topology | Attention weights messages on a support; gates decide activation under budget. |
| Need multi-λ Pareto | Fig. 10 is a frozen-policy operating curve; multi-λ retrain is optional appendix, not claimed in main text. |
| SwarmOS / PX4 contribution | Out of scope; paper contribution is algorithmic topology learning. |
'''


def renumber_figs(text: str) -> str:
    for old, new in [(14, 11), (13, 10), (12, 9), (11, 8), (10, 7), (9, 6)]:
        for a, b in [
            (f"Fig. {old}", f"@@F{new}@@"),
            (f"Fig.~{old}", f"@@T{new}@@"),
            (f"Fig.{old}", f"@@D{new}@@"),
        ]:
            text = text.replace(a, b)
    for new in (11, 10, 9, 8, 7, 6):
        text = text.replace(f"@@F{new}@@", f"Fig. {new}")
        text = text.replace(f"@@T{new}@@", f"Fig.~{new}")
        text = text.replace(f"@@D{new}@@", f"Fig.{new}")
    return text


def splice(text: str, start_marker: str, end_marker: str, insert: str) -> str:
    i = text.find(start_marker)
    j = text.find(end_marker)
    if i < 0 or j < 0 or j <= i:
        raise SystemExit(f"splice failed: {start_marker!r} -> {end_marker!r} ({i},{j})")
    return text[:i] + insert + text[j:]


def patch_en():
    t = EN.read_text(encoding="utf-8")
    t = splice(
        t,
        "**On AC-random ≥ AC-full Success.**",
        "### 5.4b Fixed Hard Top-",
        EN_INSERT,
    )
    t = t.replace(
        "**Table 2** Ablation under fixed communication interventions on a frozen AC-DSGF checkpoint (eval-only; same episode budget within this table).",
        "**Table 2** Ablation under fixed communication interventions on a frozen AC-DSGF checkpoint (eval-only; same episode budget within this table). "
        "Key: AC-random Success is slightly higher, but SCAM is **~2,600×** larger than AC-full (20.32 vs 0.0077).",
    )
    t = renumber_figs(t)
    if "## Figure Index" in t:
        t = t[: t.index("## Figure Index")] + EN_INDEX
    EN.write_text(t, encoding="utf-8")
    print("EN OK")


def patch_cn():
    t = CN.read_text(encoding="utf-8")
    t = splice(t, "**关于 AC-random ≥ AC-full Success。**", "### 5.4b", CN_INSERT)
    t = t.replace(
        "**Table 2** 在冻结 AC-DSGF checkpoint 上的固定通信干预（eval-only；表内 episode 预算一致）。",
        "**Table 2** 在冻结 AC-DSGF checkpoint 上的固定通信干预（eval-only；表内 episode 预算一致）。"
        "要点：AC-random Success 略高，但 SCAM 约为 AC-full 的 **~2,600×**（20.32 vs 0.0077）。",
    )
    t = renumber_figs(t)
    if "## 附图清单" in t:
        t = t[: t.index("## 附图清单")] + CN_INDEX
    CN.write_text(t, encoding="utf-8")
    print("CN OK")


def main():
    patch_en()
    patch_cn()
    MOCK_PATH.write_text(MOCK_TEXT, encoding="utf-8")
    print("MOCK OK")


if __name__ == "__main__":
    main()
