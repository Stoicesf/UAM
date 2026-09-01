# SECDO v2 — Final Freeze Checklist

**Status:** Phase 4 论文闭环（收敛，不扩贡献）  
**版本：** SECDO-v2.0  
**T-RO：** untouched（`paper/ac_dsgf_tro/`）

## Phase 4 总控（中文 MD）

→ **[`PHASE4_PAPER_CLOSURE.md`](PHASE4_PAPER_CLOSURE.md)**  
→ 全文结构：[`MAIN_PAPER.md`](MAIN_PAPER.md)  
→ 章节：[`sections/`](sections/)  
→ 附录：[`appendix/`](appendix/)  
→ Rebuttal：[`../rebuttal_prepare.md`](../rebuttal_prepare.md)

## Theory
- [x] Thm1 — `theory/thm1_scalar_budget_stability.md`
- [x] Lemma2 — `theory/anticipatory_projection_lemma.md`
- [x] Thm2 — `theory/thm2_prediction_aware_gap.md`
- [x] Thm3 — `theory/thm3_predictability_advantage.md`
- [x] Lemma3 — `theory/lemma3_surrogate_consistency.md`
- [x] Cor4 — `theory/corollary4_graceful_recovery.md`
- [x] Master — `theory/FINAL_THEORY_FREEZE.md`
- [x] 主文理论结构 — `sections/theory.md`
- [x] 假设层 — `appendix/A_assumptions.md`
- [x] 完整证明 — `appendix/B_proofs.md`（B.1–B.5）
- [x] 算法对齐 — `appendix/C_algorithm_details.md`
- [ ] 证明迁入 LaTeX（投稿排版阶段）

## Algorithm
- [x] Alg.1 v2 — `algorithm_secdo_v2.md` / `sections/algorithm.md`
- [x] `SECDOOptimizer` + PI / α / \(c^{\mathrm{mix}}\)

## Experiments
- [x] Synthetic / PI / Crash / UAV 多基线
- [x] Evidence chain + Fig 证据图
- [x] Intro/Results/Conclusion 对齐真实数 — `drafts/SECDO_INTRO_RESULTS_CONCLUSION.md`
- [x] Ablation A1–A3 填表（`appendix/E_additional_experiments.md` + `results/secdo_v2/ablation/`）
- [x] 论文 Fig.4（`fig4_synthetic_regret_theory.pdf`）
- [x] 论文 Fig.5（`fig5_uav_comparison.pdf`）
- [x] 附录消融图（`figE_ablation.pdf`）

## Paper integrity
- [x] 结构冻结 MD（非扩展）
- [x] Rebuttal 预答
- [x] 理论–代码一致性审计 PASS（`appendix/C4_AUDIT_REPORT.md`）
- [x] 叙事定位冻结（`NARRATIVE_POSITIONING.md`）— 非 Pareto / 非 enhanced PGD
- [x] Evidence chain 表（`tables/evidence_chain.md`）
- [x] Reviewer attack R1–R3（`reviewer_attack/`）
- [ ] `environment.yml` / `run_all.sh`（文档已约定，实现后置）
- [x] Day 8–10 LaTeX 骨架（`main.tex` + sections/appendix/figures）
- [x] `rebuttal_prepare_final.md`（10 Q）
- [x] `scripts/final_claim_check.py`
- [x] Phase 4.5 说明（`PHASE45_SUBMISSION.md` / `SUBMISSION_README.md`）
- [x] Phase 5 hardening（Complexity / Limitations / build scripts / `final_review_simulation.md`）
- [x] 版本冻结 `SECDO-v2.0-submission`

## 明确不做
Pareto 新定理 · 新网络 · Transformer · RL · 新 UAV 场景 · 平台重构
