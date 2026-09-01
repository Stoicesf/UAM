# Evidence Chain（Claim ↔ Theory ↔ Evidence）

**用途：** Appendix 或 rebuttal 开篇对照表；禁止增列未做实验。  
**叙事权威：** [`../NARRATIVE_POSITIONING.md`](../NARRATIVE_POSITIONING.md)

---

| Claim | Theory | Evidence |
|-------|--------|----------|
| 学习约束动力学使集合误差可控 | **Thm.1** \(d_H\le\delta\)（及 \(L_g\epsilon\)） | Synthetic / UAV \(\delta\) 日志；训练 \(L_c\)（Assump.5 / Lem.1） |
| 投影扰动受 \(C_\Pi\delta\) 控制 | **Lem.2** + Assump.3 | Oracle（真 \(c_{t+1}\)）与 SECDO 的 gap/viol 差距；同 \(y_t\) 几何 |
| 动态遗憾对漂移路径变差敏感 | **Thm.2** \(O\sqrt{T(1+P_T)}+O\sum(\epsilon+\delta)\) | **Fig.4**（尺度验证）；UAV Table I：\(\bar\chi\times13\Rightarrow\mathrm{Reg}_T\times56.7\) |
| \(\delta<\chi\)（\(\mathrm{PI}<1\)）时提前投影证书更紧 | **Thm.3** | PI boundary；Fig.3（\(PI\uparrow\Rightarrow\alpha\downarrow\)） |
| 预测失效时优雅降级，非永不失败 | **Cor.4** | **Fig.2** crash；窗内累积违规 |
| 自适应 \(\alpha\) 是失效遏制，非刷名次 | Cor.4 + 设计 | **Ablation A3**：干净 Fast 遗憾≈Full；crash 窗内上升 ×3 |
| 提前性降低遗憾 | Thm.3 方向 | **Ablation A1/A2**：Fast \(\mathrm{Reg}_T\) +18% |
| 系统层 optimality–safety 权衡 | 定位（非 Pareto） | **Fig.5**：SECDO gap↓、violation 相对 reactive 略↑；Oracle 为上界 |

---

## 证据 ID 速查（落盘）

| 代号 | 路径 |
|------|------|
| Fig.2 | `results/secdo_v2/paper_figures/fig3_cor4_graceful_degradation.pdf` |
| Fig.3 | `…/fig2_thm3_pi_alpha.pdf` |
| Fig.4 | `…/fig4_synthetic_regret_theory.pdf` + `fig4_theory_bound_meta.json` |
| Fig.5 | `…/fig5_uav_comparison.pdf` |
| Fig.E | `…/figE_ablation.pdf` |
| Table I | `EVIDENCE_CHAIN.json` / `drafts/SECDO_INTRO_RESULTS_CONCLUSION.md` |
| Ablation | `results/secdo_v2/ablation/summary.json` |

---

## LaTeX 草稿（Appendix）

```latex
\begin{table}[t]
\centering
\caption{Evidence chain linking claims, theory, and experiments.}
\label{tab:evidence_chain}
\begin{tabular}{p{0.32\linewidth}p{0.22\linewidth}p{0.36\linewidth}}
\toprule
Claim & Theory & Evidence \\
\midrule
Constraint error controlled by prediction & Thm.\,1 & $\delta$ logs / $L_c$ \\
Projection perturbation bound & Lem.\,2 & Oracle--SECDO gap \\
Dynamic regret drift sensitivity & Thm.\,2 & Fig.\,4; Table\,I \\
Tighter certificate iff $\delta<\chi$ & Thm.\,3 & Fig.\,3; PI boundary \\
Graceful degradation & Cor.\,4 & Fig.\,2 (crash) \\
Adaptive $\alpha$ = failure containment & Cor.\,4 & Ablation A3 \\
\bottomrule
\end{tabular}
\end{table}
```
