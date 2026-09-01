# -*- coding: utf-8 -*-
"""Insert Corollary 1–2 into EN/CN MD manuscripts (UTF-8 safe)."""
from pathlib import Path

EN = Path(r"f:\UAM\paper\ac_dsgf\AC_DSGF_EN.md")
CN = Path(r"f:\UAM\paper\ac_dsgf_cn\AC_DSGF_CN.md")

COR1_EN = r"""
**Corollary 1 (Task-Return Degradation Bound).** Under the Lipschitz conditions of Prop. 2, let the stage reward \(r_t\) be \(L_r\)-Lipschitz in the agent action \(a_t\). Denote \(J_{\mathrm{full}} = \mathbb{E}[\sum_t r_t(a_t^\star)]\) as the expected return under full (reference) communication, and \(J_{\mathrm{sparse}}\) under sparse topology with residual correction. Then:
\[
|J_{\mathrm{full}} - J_{\mathrm{sparse}}|
\;\le\;
L_r \cdot \beta L_\pi \cdot \varepsilon \cdot T,
\]
where \(\varepsilon = \max_t \|m_t^\star - m_t\|\) is the maximum message deviation over the horizon, and \(T\) is the episode length.

*Interpretation.* This moves the stability analysis from *action-space deviation* to *cumulative task reward*. It shows that the performance gap is linearly bounded by the message approximation error \(\varepsilon\) and the residual correction gain \(\beta\). It does **not** claim that sparsity improves return; rather, it provides a quantitative condition under which aggressive SCAM reduction does not catastrophically degrade task performance—consistent with the graceful degradation observed in Table 2 and Sec. 5.5.

"""

COR2_EN = r"""
**Corollary 2 (Top-K Selection under Bounded Utility Estimation).** Let the true marginal utility of activating edge \((i,j)\) be \(u_{ij} \in [0,1]\), and let the learned score be \(\hat{s}_{ij}\) with estimation error \(\|\hat{s}_{ij} - u_{ij}\|_\infty \le \delta\). For a per-agent budget \(K\), let \(E_{\mathrm{opt}}\) be the edge set that maximizes \(\sum u_{ij}\) subject to \(|E_i|\le K\), and \(E_{\mathrm{top}}\) be the set selected by ranking \(\hat{s}_{ij}\). Then the utility gap satisfies:
\[
\sum_{E_{\mathrm{opt}}} u_{ij}
-
\sum_{E_{\mathrm{top}}} u_{ij}
\;\le\;
2K\delta.
\]

*Interpretation.* This provides a theoretical grounding for the Top-\(K\) deployment results in Fig. 9: as long as the learned scores approximate the true task utility within a bounded error \(\delta\), ranking by \(\hat{s}\) yields a hard topology whose utility is within \(\mathcal{O}(K\delta)\) of the optimal \(K\)-neighbor selection. It **does not** claim that Top-\(K\) training is better than soft training; rather, it explains why soft-learned scores can be discretized into competitive hard links at inference without retraining.

"""

COR1_CN = r"""
**推论 1（任务回报退化界）。** 在命题 2 的 Lipschitz 条件下，设单步奖励 \(r_t\) 对动作 \(a_t\) 为 \(L_r\)-Lipschitz。记 \(J_{\mathrm{full}}=\mathbb{E}[\sum_t r_t(a_t^\star)]\) 为充分通信下的期望回报，\(J_{\mathrm{sparse}}\) 为稀疏拓扑+残差修正下的回报，则：
\[
|J_{\mathrm{full}} - J_{\mathrm{sparse}}|
\;\le\;
L_r \cdot \beta L_\pi \cdot \varepsilon \cdot T,
\]
其中 \(\varepsilon = \max_t \|m_t^\star - m_t\|\) 为时域内最大消息偏差，\(T\) 为 episode 长度。

*解释.* 该推论将稳定性分析从*动作空间偏差*提升至*累积任务回报*。它给出性能差距被消息近似误差 \(\varepsilon\) 与残差增益 \(\beta\) 线性上界的条件，**不**声称稀疏化提升回报；而是给出 SCAM 大幅降低时不至于灾难性性能退化的定量条件——与表 2 及 §5.5 平缓退化一致。

"""

COR2_CN = r"""
**推论 2（有界效用估计下的 Top-K 选择）。** 设激活边 \((i,j)\) 的真实边际效用为 \(u_{ij}\in[0,1]\)，所学分数为 \(\hat{s}_{ij}\)，且估计误差 \(\|\hat{s}_{ij} - u_{ij}\|_\infty \le \delta\)。对每智能体预算 \(K\)，记 \(E_{\mathrm{opt}}\) 为最大化 \(\sum u_{ij}\) 且满足 \(|E_i|\le K\) 的最优边集，\(E_{\mathrm{top}}\) 为按 \(\hat{s}_{ij}\) 排序选择的边集。则效用差距满足：
\[
\sum_{E_{\mathrm{opt}}} u_{ij}
-
\sum_{E_{\mathrm{top}}} u_{ij}
\;\le\;
2K\delta.
\]

*解释.* 该推论为 Fig. 9 的 Top-\(K\) 部署提供理论支撑：只要所学分数在误差 \(\delta\) 内有界，按 \(\hat{s}\) 排序得到的硬拓扑效用与最优 \(K\) 近邻选择差距为 \(\mathcal{O}(K\delta)\)。**不**声称 Top-\(K\) 训练优于软训练；而是解释为什么软学分数在推理阶段可离散化为有竞争力的硬链路。

"""


def insert_after_marker(text: str, marker: str, block: str, already: str) -> str:
    if already in text:
        print("skip already present:", already[:40])
        return text
    i = text.find(marker)
    if i < 0:
        raise SystemExit(f"marker not found: {marker[:60]!r}")
    # find end of Prop.2 / Prop.3 section: next #### or ### 4.7
    # Insert after the interpretation paragraph of the proposition, before next #### or ###
    # Use marker as start of prop header; find next section header after marker
    j = text.find("\n#### ", i + len(marker))
    k = text.find("\n### ", i + len(marker))
    candidates = [x for x in (j, k) if x >= 0]
    if not candidates:
        raise SystemExit("no next section after " + marker[:40])
    end = min(candidates)
    return text[:end] + "\n" + block + text[end:]


def patch_en():
    t = EN.read_text(encoding="utf-8")
    # Insert Cor1 after Prop.2 block (before Prop.3)
    t = insert_after_marker(
        t,
        "#### Prop.2 — Residual Stability Analysis",
        COR1_EN,
        "Corollary 1 (Task-Return Degradation Bound)",
    )
    # Insert Cor2 after Prop.3 (before ### 4.7)
    t = insert_after_marker(
        t,
        "#### Prop.3 — Interpretation of Adaptive Topology Learning",
        COR2_EN,
        "Corollary 2 (Top-K Selection under Bounded Utility Estimation)",
    )
    # Experiment citations
    old_deg = "~11% relative drop under a 90% budget cut (graceful degradation). Failure boundary:"
    new_deg = (
        "~11% relative drop under a 90% budget cut (graceful degradation). "
        "The graceful degradation aligns with Corollary 1 under small message deviation \(\\varepsilon\). "
        "Failure boundary:"
    )
    if "aligns with Corollary 1" not in t:
        t = t.replace(old_deg, new_deg)

    old_fig9 = (
        "**Fig. 9** Soft learned topology can be discretized into practical communication links "
        "at comparable task effectiveness. Mild gains after removing weak soft activations indicate "
        "unnecessary interactions at inference; we do **not** claim Top-\\(K\\) should replace soft training."
    )
    new_fig9 = (
        "**Fig. 9** Soft learned topology can be discretized into practical communication links "
        "at comparable task effectiveness. This empirical observation is theoretically grounded by "
        "Corollary 2. Mild gains after removing weak soft activations indicate unnecessary "
        "interactions at inference; we do **not** claim Top-\\(K\\) should replace soft training."
    )
    if "grounded by Corollary 2" not in t:
        t = t.replace(old_fig9, new_fig9)

    EN.write_text(t, encoding="utf-8")
    print("EN OK", "Corollary 1" in t, "Corollary 2" in t)


def patch_cn():
    t = CN.read_text(encoding="utf-8")
    t = insert_after_marker(
        t,
        "#### Prop.2 — Residual Stability Analysis",
        COR1_CN,
        "推论 1（任务回报退化界）",
    )
    t = insert_after_marker(
        t,
        "#### Prop.3 — Interpretation of Adaptive Topology Learning",
        COR2_CN,
        "推论 2（有界效用估计下的 Top-K 选择）",
    )
    old_deg = "90% 预算削减下约 11% 相对下降（平缓退化）。失败边界："
    new_deg = (
        "90% 预算削减下约 11% 相对下降（平缓退化）。"
        "该平缓退化与推论 1 在消息偏差 \(\\varepsilon\) 较小时一致。"
        "失败边界："
    )
    if "推论 1" in t and "平缓退化与推论 1" not in t:
        t = t.replace(old_deg, new_deg)

    old_fig9 = (
        "**Fig. 9** 软所学拓扑可离散化为实际通信链路，任务有效性相当。"
        "去除弱软激活后的温和增益表明推理阶段存在不必要交互；"
        "我们**不**主张 Top-\\(K\\) 应取代软训练。"
    )
    new_fig9 = (
        "**Fig. 9** 软所学拓扑可离散化为实际通信链路，任务有效性相当。"
        "该经验观察由推论 2 提供理论支撑。"
        "去除弱软激活后的温和增益表明推理阶段存在不必要交互；"
        "我们**不**主张 Top-\\(K\\) 应取代软训练。"
    )
    if "推论 2 提供理论支撑" not in t:
        t = t.replace(old_fig9, new_fig9)

    CN.write_text(t, encoding="utf-8")
    print("CN OK", "推论 1" in t, "推论 2" in t)


if __name__ == "__main__":
    patch_en()
    patch_cn()
