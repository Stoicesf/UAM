# SECDO-v2.0 预审稿 / Rebuttal 材料

**用途：** Day 7 模拟审稿 + 投稿后快速回复  
**原则：** 只引用已冻结理论与已落盘证据；不承诺未做实验。  
**深度攻击预演：** `paper/ac_dsgf_v2/reviewer_attack/R{1,2,3}_*.md`  
**定位：** 非全面 Pareto；opt–safety 权衡 + failure containment。

---

## Q1. Is SECDO just prediction + projected gradient?

**短答：** No.

**展开：**

- 预测误差 \(\delta_t\) **进入**优化保证（Thm.1–2），而非外挂预报模块。  
- 优势由 \(\mathrm{PI}_t=\delta_t/\chi_t\) **条件化**（Thm.3）：仅当 \(\delta<\chi\) 时提前投影给出更紧证书。  
- 决策使用 \(c^{\mathrm{mix}}=\alpha\hat c+(1-\alpha)c\)，\(\alpha=1/(1+\mathrm{PI}^2)\)，是连续插值，不是 “forecast-then-PGD” 硬切换。

**可附图：** Fig.3（PI–\(\alpha\) 轨迹）。

---

## Q2. Why not always use the oracle?

**短答：** Oracle 需要未来可行域 \(c_{t+1}\)；SECDO 在线学习 \(\hat{\mathcal{B}}_{t+1}\)，无未来访问。

**展开：**

- Oracle 是性能上界参照，不是可部署算法。  
- SECDO 的价值在于：用可学习的 \(\hat c\) 逼近 oracle 证书，并在 \(\mathrm{PI}>1\) 时自动回退。

**数据锚：** UAV 表中 Oracle 违规最低；SECDO 在无未来信息下保持违规 <2%。

---

## Q3. What if prediction fails?

**短答：** Corollary 4 — graceful degradation，不是 failure avoidance。

**展开：**

- \(\mathrm{PI}\uparrow\Rightarrow\alpha\downarrow\)，混合预算靠近 \(c_t\)（反应安全网）。  
- 示意界：\(\mathrm{Reg}_{\mathrm{SECDO}}\le\mathrm{Reg}_R+O(\sum\mathbf{1}^{\mathrm{fail}}\delta)\)（以冻结 Cor4 为准）。  
- **实验：** crash protocol（论文 Fig.2），对比 \(\alpha\equiv1\)；**不要**把 UAV Stress 误说成唯一 Cor4 证据。

---

## Q4. Does the \(56.7\times\) regret prove the \(\sqrt{T(1+\sum\chi)}\) rate exactly?

**短答：** 证明的是**漂移敏感性与上界形态一致**，不是精确幂律拟合。

**展开：**

- Thm.2 给出含 \(P_T\)（受 \(\sum\chi\) 影响）的上界。  
- 固定 \(T\) 下，\(\bar\chi\times13\) 伴随 \(\mathrm{Reg}_T\times56.7\)，说明遗憾对漂移的响应可**显著强于线性**。  
- 不声称观测比等于 \(\sqrt{13}\approx3.6\)；完整 rate 验证以 synthetic + 理论曲线（Fig.4）为主。

---

## Q5. Which component matters?（消融 · 已落地）

数据：`results/secdo_v2/ablation/summary.json` · 表见 `appendix/E_additional_experiments.md`

| 去掉 | 实测后果 |
|------|----------|
| \(\hat c\) / \(\Pi_{\hat B}\)（A1≡A2） | Fast \(\mathrm{Reg}_T\): 0.044→0.052（+18%） |
| 自适应 \(\alpha\)（A3） | Crash 窗内累积上升：0.185→0.559（×3.0）；干净 Fast 上 A3 遗憾可略优，故 Cor4 叙事以 crash 为准 |

正式回复一句：anticipation 降遗憾；adaptive \(\alpha\) 在预测崩溃时防发散。

---

## Q6. Relation to DSGF / AC-DSGF / T-RO track?

**短答：** 同应用域基线；本文贡献在约束动力学 + PAP，不是重投 T-RO 文稿。

**红线：** 不修改、不混写 `paper/ac_dsgf_tro/`。

---

## Q7. ML reviewer: Does minimizing \(L_c\) solve the control problem?

**短答：** \(L_c\) 是 \(\delta\) 的可训练代理（Lemma 1 / surrogate consistency），经 Thm.1–2 进入证书；不是直接最小化 \(\mathrm{Reg}_T\)。

---

## Q8. Robotics reviewer: Is 1.89% violation under Stress acceptable?

**短答：** Stress 为最强漂移；违规相对 Slow 上升但仍有界，且无崩溃；与 Cor4 叙事一致。绝对值依赖教师尺度，应报告相对基线（Oracle / Reactive）而非单一绝对阈值。

---

## 模拟审稿角色清单（Day 7）

| 角色 | 焦点 | 对应 Q |
|------|------|--------|
| Optimization | 界的假设与 tightness | Q1, Q4 |
| ML | \(L_c\)、泛化、消融 | Q5, Q7 |
| Robotics / systems | 违规、崩溃、可部署 | Q2, Q3, Q8 |
