# -*- coding: utf-8 -*-
"""P0–P3 manuscript polish: SCAM→SCA, DSGF budget note, theory/exp wording."""
from pathlib import Path

EN = Path(r"f:\UAM\paper\ac_dsgf\AC_DSGF_EN.md")
CN = Path(r"f:\UAM\paper\ac_dsgf_cn\AC_DSGF_CN.md")


def rename_sca(t: str) -> str:
    reps = [
        ("Soft Communication Activation Mass (SCAM)", "Soft Communication Activation (SCA)"),
        ("Soft Communication Activation Mass", "Soft Communication Activation"),
        ("Soft Comm. Mass", "SCA"),
        ("SCAM", "SCA"),
    ]
    for a, b in reps:
        t = t.replace(a, b)
    return t


def patch_en():
    t = EN.read_text(encoding="utf-8")
    t = t.replace(
        "# Learning Adaptive Communication Topologies for Communication-Constrained UAV Swarm Coordination",
        "# Learning Adaptive Topologies for Communication-Constrained UAV Swarm Coordination",
    )
    t = rename_sca(t)

    # section header after rename
    t = t.replace(
        "### 4.5 Soft Communication Activation (SCA)",
        "### 4.5 Soft Communication Activation (SCA)",
    )
    # definition paragraph polish
    t = t.replace(
        "We define **Soft Communication Activation (SCA)**\n\\[\nC_s=\\sum_{i,j}g_{ij}\n\\]\nas a *training / analysis proxy* for continuous activation intensity.  \nSCA is **not** a radio packet count.",
        "We define **Soft Communication Activation (SCA)**\n\\[\nC_s=\\sum_{i,j}g_{ij}\n\\]\nas a *training / analysis proxy* for continuous activation intensity (activation mass).  \nSCA is **not** a radio packet count.",
    )

    t = t.replace(
        "We do **not** claim that the realized adaptive edge count equals \\(KN\\) with equality.",
        "We do **not** claim that the realized edge count exactly equals \\(KN\\).",
    )

    # Abstract end already has SCA after rename; ensure wording
    t = t.replace(
        "while inducing substantially lower Soft Communication Activation (SCA) across swarm scales.",
        "while maintaining substantially lower Soft Communication Activation (SCA) across swarm scales.",
    )

    # Corollary 1 interpretation: already says consistent - fine
    # §5.5 soften
    t = t.replace(
        "~11% relative drop under a 90% budget cut (graceful degradation). The graceful degradation aligns with Corollary 1 under small message deviation \\(\\varepsilon\\). Failure boundary:",
        "~11% relative drop under a 90% budget cut (graceful degradation). "
        "This graceful degradation is **consistent with the spirit of Corollary 1**: "
        "action deviation caused by sparsification remains bounded, as learned gates retain "
        "task-critical edges even under aggressive budget cuts. Failure boundary:",
    )

    # DSGF flat budget note after the rho table - find and insert
    dsgf_note = (
        "\n\n> **Note on DSGF under budget masks.** DSGF does not employ a learnable gate; "
        "its effective edges are determined by a fixed radius. Applying post-hoc Top-\\(K\\)/budget "
        "masks to DSGF yields negligible Success change because its active neighborhood count "
        "already falls in a saturated regime relative to the aggressive 10% budget cut in this "
        "dense swarm setting (i.e., effective \\(k\\) is already small compared with the opened "
        "support). The flat DSGF curve therefore reflects a **saturated budget region for DSGF**, "
        "not a claim that DSGF violates the evaluation mask. AC-DSGF, by contrast, remains "
        "sensitive to the same budget schedule while keeping much lower SCA.\n"
    )
    if "Note on DSGF under budget masks" not in t:
        marker = "| 10% | 16.9 | 21.7 | 24.2 |\n"
        if marker in t:
            t = t.replace(marker, marker + dsgf_note)
        else:
            print("WARN: DSGF note anchor missing")

    # Residual N=4
    if "isolate the effect" not in t:
        t = t.replace(
            "| w/o Residual (\\(N{=}4\\)) | —| —| × | 0.22 | —|\n| Full DSGF (\\(N{=}4\\)) | — | — | ✓ | 9.27 | — |",
            "| w/o Residual (\\(N{=}4\\)) | — | — | × | 0.22 | — |\n"
            "| Full DSGF (\\(N{=}4\\)) | — | — | ✓ | 9.27 | — |\n\n"
            "*Residual ablation is a **mechanism check** in a smaller \\(N{=}4\\) setting to isolate "
            "silence collapse (9.27%→0.22%); the same residual module is used in the \\(N{=}16\\) "
            "main results and should not be over-extrapolated as an \\(N{=}16\\) Success claim.*\n",
        )
        # try alternate dashes
        if "isolate the effect" not in t and "mechanism check" not in t:
            old = "| w/o Residual (\\(N{=}4\\)) |"
            idx = t.find(old)
            if idx >= 0:
                # insert after Full DSGF N=4 line
                j = t.find("\n", t.find("Full DSGF (\\(N{=}4\\))", idx))
                if j > 0:
                    note = (
                        "\n\n*Residual ablation is a **mechanism check** in a smaller \\(N{=}4\\) "
                        "setting to isolate silence collapse (9.27%→0.22%); the same residual module "
                        "is used in the \\(N{=}16\\) main results and should not be over-extrapolated "
                        "as an \\(N{=}16\\) Success claim.*\n"
                    )
                    t = t[: j + 1] + note + t[j + 1 :]

    # Behavior / proximity defense
    prox = (
        "SCA correlates with proximity risk (~+0.84) and dispersion (~−0.57)—risk-conditioned "
        "sparsity, not fixed-rate chatter.\n\n"
        "In the navigation task, collision risk is the dominant coordination factor, hence the "
        "high proximity correlation. However, AC-DSGF’s asymmetry (directional gating; cf. "
        "Table 2b) and task-phase stability (Fig. 6) indicate a **structured, non-reciprocal** "
        "sparsity pattern beyond mere distance decay—e.g., informative leader–follower links "
        "can persist as relative distances vary, whereas a pure distance heuristic would toggle "
        "more symmetrically.\n"
    )
    old_prox = (
        "SCA correlates with proximity risk (~+0.84) and dispersion (~−0.57)—risk-conditioned "
        "sparsity, not fixed-rate chatter.\n"
    )
    if "non-reciprocal" not in t and old_prox in t:
        t = t.replace(old_prox, prox)

    # Table 5 expand with Success + protocol
    old_t5 = """**Table 5** SCA \\(C_s\\) and \\(\\eta_N=C_s/(N(N-1))\\)

| \\(N\\) | DSGF \\(C\\) | DSGF \\(\\eta_N\\) | AC \\(C\\) | AC \\(\\eta_N\\) |
|------|----------:|---------------:|--------:|-------------:|"""
    # read actual table from file after rename - flexible replace
    import re

    m = re.search(
        r"\*\*Table 5\*\*.*?\n\n\| \\\(N\\\) \|.*?\n\|------\|.*?\n(?:\|.*?\n){3}",
        t,
        flags=re.S,
    )
    new_t5 = """**Table 5** SCA \\(C_s\\), normalized density \\(\\eta_N=C_s/(N(N-1))\\), and zero-shot Success (%).
*Protocol note:* 64 episodes, seed 42 (eval-only). SCA magnitudes differ from Table 1 within stochastic variance of seed/rollout sets; the key trend \\(\\eta_N=\\mathcal{O}(1/N)\\) is preserved. Success at \\(N{=}32\\) remains nonzero for both methods—AC trades a small Success drop for orders-of-magnitude lower SCA.

| \\(N\\) | DSGF SCA | DSGF \\(\\eta_N\\) | DSGF Succ. | AC SCA | AC \\(\\eta_N\\) | AC Succ. |
|------|--------:|---------------:|-----------:|-------:|-------------:|---------:|
| 8 | 11.01 | 0.197 | 33.3 | 0.0020 | \\(3.6\\times10^{-5}\\) | **40.6** |
| 16 | 22.62 | 0.094 | 19.8 | 0.0059 | \\(2.5\\times10^{-5}\\) | **21.1** |
| 32 | 56.06 | 0.057 | 11.2 | 0.0188 | \\(1.9\\times10^{-5}\\) | 10.3 |
"""
    if m:
        t = t[: m.start()] + new_t5 + t[m.end() :]
        print("Table 5 replaced")
    else:
        print("WARN Table 5 pattern not found")
        # fallback: just add protocol after Table 5 header
        t = t.replace(
            "**Table 5** SCA \\(C_s\\) and \\(\\eta_N=C_s/(N(N-1))\\)",
            "**Table 5** SCA \\(C_s\\), \\(\\eta_N\\), and zero-shot Success. "
            "*Protocol: 64 episodes, seed 42; SCA vs Table 1 differs within rollout variance; "
            "\\(\\eta_N=\\mathcal{O}(1/N)\\) trend preserved.*",
        )

    # Fig index Success—SCA already after rename
    EN.write_text(t, encoding="utf-8")
    print("EN done", t.count("SCAM"), "SCAM left;", t.count("SCA"), "SCA")


def patch_cn():
    t = CN.read_text(encoding="utf-8")
    t = t.replace(
        "# 面向通信约束多无人机蜂群协同的自适应通信拓扑学习",
        "# 面向通信约束多无人机蜂群协同的自适应拓扑学习",
    )
    t = rename_sca(t)
    t = t.replace(
        "诱导显著更低的 Soft Communication Activation (SCA)",
        "保持显著更低的 Soft Communication Activation (SCA)",
    )
    t = t.replace(
        "我们**不**声称自适应边数恒等于 \\(KN\\)。",
        "我们**不**声称实现边数精确等于 \\(KN\\)。",
    )
    # also English leftover
    t = t.replace(
        "We do **not** claim that the realized adaptive edge count equals \\(KN\\) with equality.",
        "We do **not** claim that the realized edge count exactly equals \\(KN\\).",
    )

    t = t.replace(
        "该平缓退化与推论 1 在消息偏差 \\(\\varepsilon\\) 较小时一致。失败边界：",
        "该平缓退化与**推论 1 的精神一致**：稀疏化引起的动作偏差仍保持有界，因为所学门控在激进预算削减下仍保留任务关键边。失败边界：",
    )

    dsgf_note = (
        "\n\n> **关于 DSGF 在预算扫描下的平坦曲线。** DSGF 无学习门控，有效边由固定半径决定。"
        "对其施加事后 Top-\\(K\\)/预算掩码时 Success 变化很小，因为在本稠密蜂群设定下，其有效邻域规模相对激进的 10% 预算削减已处于饱和区。"
        "平坦曲线反映的是 **DSGF 的预算饱和区**，而非 DSGF 未遵守评估掩码。AC-DSGF 对同一预算日程仍敏感，同时保持低得多的 SCA。\n"
    )
    if "关于 DSGF 在预算扫描" not in t:
        marker = "| 10% | 16.9 | 21.7 | 24.2 |\n"
        if marker in t:
            t = t.replace(marker, marker + dsgf_note)

    if "机制验证" not in t:
        idx = t.find("w/o Residual")
        if idx > 0:
            j = t.find("Full DSGF", idx)
            j = t.find("\n", j)
            note = (
                "\n\n*残差消融是在较小 \\(N{=}4\\) 设定下的**机制验证**，用以隔离静默坍塌"
                "（9.27%→0.22%）；\\(N{=}16\\) 主结果使用同一残差模块，不宜过度外推为 \\(N{=}16\\) 的 Success 主张。*\n"
            )
            t = t[: j + 1] + note + t[j + 1 :]

    old_prox = (
        "SCA 与邻近风险正相关（约 +0.84）、与分散度负相关（约 −0.57）——风险条件化稀疏，而非固定速率通信。\n"
    )
    prox = (
        "SCA 与邻近风险正相关（约 +0.84）、与分散度负相关（约 −0.57）——风险条件化稀疏，而非固定速率通信。\n\n"
        "在导航任务中，碰撞风险是协同的主导因素，故邻近相关较高。然而，AC-DSGF 的非对称性（有向门控；见表 2b）"
        "与任务阶段稳定性（Fig. 6）表明其学到的是**结构化、非互易**的稀疏模式，而非单纯距离衰减——"
        "例如信息性的领导—跟随链路可在相对距离变化时保持，而纯距离启发式更倾向于对称切换。\n"
    )
    if "非互易" not in t and old_prox in t:
        t = t.replace(old_prox, prox)

    import re

    m = re.search(
        r"\*\*Table 5\*\*.*?\n\n\| \\\(N\\\) \|.*?\n\|------\|.*?\n(?:\|.*?\n){3}",
        t,
        flags=re.S,
    )
    new_t5 = """**Table 5** SCA \\(C_s\\)、归一化密度 \\(\\eta_N\\) 与零样本 Success（%）。
*协议说明：* 64 episodes，seed 42（仅评估）。SCA 量级与表 1 因种子/rollout 随机性而有差异，但关键趋势 \\(\\eta_N=\\mathcal{O}(1/N)\\) 保持。\\(N{=}32\\) 时两方法 Success 仍非零——AC 以小幅 Success 代价换取数量级更低的 SCA。

| \\(N\\) | DSGF SCA | DSGF \\(\\eta_N\\) | DSGF Succ. | AC SCA | AC \\(\\eta_N\\) | AC Succ. |
|------|--------:|---------------:|-----------:|-------:|-------------:|---------:|
| 8 | 11.01 | 0.197 | 33.3 | 0.0020 | \\(3.6\\times10^{-5}\\) | **40.6** |
| 16 | 22.62 | 0.094 | 19.8 | 0.0059 | \\(2.5\\times10^{-5}\\) | **21.1** |
| 32 | 56.06 | 0.057 | 11.2 | 0.0188 | \\(1.9\\times10^{-5}\\) | 10.3 |
"""
    if m:
        t = t[: m.start()] + new_t5 + t[m.end() :]
        print("CN Table 5 replaced")
    else:
        print("WARN CN Table 5 not found")

    CN.write_text(t, encoding="utf-8")
    print("CN done", t.count("SCAM"), "SCAM left;", "SCA" in t)


def patch_aux():
    for p in [
        Path(r"f:\UAM\paper\ac_dsgf\LANGUAGE_FREEZE.md"),
        Path(r"f:\UAM\paper\ac_dsgf\HIGHLIGHTS.md"),
        Path(r"f:\UAM\paper\ac_dsgf\HIGHLIGHTS_CN.md"),
        Path(r"f:\UAM\paper\ac_dsgf\CONTRIBUTION_STATEMENT.md"),
        Path(r"f:\UAM\paper\ac_dsgf\CONTRIBUTION_STATEMENT_CN.md"),
        Path(r"f:\UAM\paper\ac_dsgf\COVER_LETTER.md"),
        Path(r"f:\UAM\paper\ac_dsgf\COVER_LETTER_CN.md"),
        Path(r"f:\UAM\paper\ac_dsgf\response_mock.md"),
        Path(r"f:\UAM\paper\ac_dsgf\THEORY_EVIDENCE_ALIGNMENT.md"),
        Path(r"f:\UAM\paper\ac_dsgf\RAL_SUBMIT_CHECKLIST.md"),
        Path(r"f:\UAM\paper\ac_dsgf\MANUSCRIPT_INDEX.md"),
        Path(r"f:\UAM\paper\ac_dsgf\AC_DSGF_v1_submission_checklist.md"),
    ]:
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        t2 = rename_sca(t)
        t2 = t2.replace(
            "Learning Adaptive Communication Topologies for Communication-Constrained UAV Swarm Coordination",
            "Learning Adaptive Topologies for Communication-Constrained UAV Swarm Coordination",
        )
        t2 = t2.replace(
            "面向通信约束多无人机蜂群协同的自适应通信拓扑学习",
            "面向通信约束多无人机蜂群协同的自适应拓扑学习",
        )
        if "SCAM" in t2 or "禁止 SCAM" not in t2:
            # add freeze note once
            if p.name == "LANGUAGE_FREEZE.md" and "禁止使用 SCAM" not in t2:
                t2 += "\n\n## 术语禁忌\n- **禁止使用 SCAM**（英语俚语“骗局”联想）。统一：**SCA** = Soft Communication Activation，\\(C_s=\\sum g_{ij}\\)。\n"
        p.write_text(t2, encoding="utf-8")
        print("aux", p.name, "SCAM left", t2.count("SCAM"))


if __name__ == "__main__":
    patch_en()
    patch_cn()
    patch_aux()
