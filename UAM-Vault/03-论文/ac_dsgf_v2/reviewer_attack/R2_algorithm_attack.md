# R2 — Algorithm / ML Reviewer Attack Simulation

**角色：** 算法与机器学习审稿人  
**目标：** 堵死 “只是 PGD+预测头” 降级叙事  
**定位：** predictive **constraint evolution** + predictability-conditioned projection + failure-safe \(\alpha\)

---

## B1. “Is SECDO just prediction + projected gradient?”

**攻击：** 贡献是工程拼装。

**回复：**

1. 预测对象是 **约束 / 可行域**（\(\hat c\to\hat{\mathcal{B}}\)），不是仅状态或奖励。  
2. \(\delta_t\) **进入** Thm.1–2 优化保证，而非外挂预报。  
3. 决策经 \(\mathrm{PI}_t\) **条件化**混合 \(c^{\mathrm{mix}}\)，不是 forecast-then-PGD 硬切换。  
4. 训练 \(L_c\) 与在线 Algorithm 1 **分离**（Training ≠ Online）。

**禁止自贬：** “prediction-enhanced PGD”。  
**正确自称：** “predictive constraint-evolution framework for dynamic feasible optimization”。

---

## B2. “Why not always use oracle / peek \(c_{t+1}\)?”

**回复：** Oracle 需要未来可行域，不可部署；SECDO 学习 \(\hat{\mathcal{B}}_{t+1}\)。Fig.5 中 Oracle 为上界参照，用于量度 gap，不是基线可选项。

---

## B3. “Why this \(\alpha=1/(1+\mathrm{PI}^2)\)? Why not threshold switch?”

**回复：**

- 连续、可微、\(\mathrm{PI}\to\infty\Rightarrow\alpha\to0\)，满足 Cor.4 退化。  
- 硬开关在 \(\mathrm{PI}\approx1\) 处引入抖动；连续插值对应 “conditioned optimization” 叙事。  
- Ablation A3 证明：固定 \(\alpha=1\) 在失效时放大违规 → 自适应必要。

不声称该公式是唯一最优调度，声称它是 **failure-safe** 的充分机制。

---

## B4. “Why predict constraints rather than states only?”

**回复：** Thm.1 将容量误差直接译为 \(d_H\)；状态误差经 Assump.1 Lipschitz 再进 \(\delta\)。约束头使理论项与损失 \(L_c\) 对齐（Lem.1 surrogate consistency）——这是 ML reviewer 关心的桥。

---

## B5. “Minimizing \(L_c\) solves the control problem?”

**回复：** **否。** \(L_c\) 收缩 \(\delta\)；\(\mathrm{Reg}_T\) 另含漂移 \(P_T\) 与优化动力学。端到端不宣称最优控制。

---

## B6. “Detach \(\hat c\) through \(\Pi\) 是否切断学习？”

**回复：** 训练稳定性约定；约束头仍由 \(L_c\) 监督。投影路径不回传避免病态 Jacobian。在线阶段可选小步更新 \(\theta\)（trust-region），与 Algorithm 1 推理分离。

---

## B7. “DSGF / AC-DSGF 只是 myopic adapter？”

**诚实回复：** 当前平台在同一标量预算教师下比较；经典拓扑耦合未完全展开。主文写 *demonstrate effectiveness under shared teacher*，不宣称击败完整 DSGF 族所有变体。贡献在约束演化框架，不在重实现通信图算法。

---

## R2 检查清单

- [ ] Intro / Abstract 无 “enhanced PGD”  
- [ ] Algorithm 节强调 PI–\(\alpha\)–\(c^{\mathrm{mix}}\)  
- [ ] Training ≠ Online 在 appendix C 可见  
