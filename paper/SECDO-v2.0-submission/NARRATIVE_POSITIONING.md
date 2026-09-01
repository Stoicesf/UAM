# SECDO 投稿叙事定位（冻结）

**Status:** Phase 4 Day 5–7 · 叙事收敛  
**一句话：** SECDO **不是**追求所有指标 Pareto 最优，而是在动态约束环境中，通过预测降低 optimality gap，并通过自适应投影控制安全风险。

---

## 最终定位（唯一合法）

> **A predictive constraint-evolution framework for dynamic feasible optimization.**

### 三关键词（全文统一）

| # | 关键词 | 是 | 不是 |
|---|--------|----|------|
| 1 | **Constraint evolution** | 学习可行域 \(\mathcal{B}_t\to\hat{\mathcal{B}}_{t+1}\) | 纯 state prediction / world-model 噱头 |
| 2 | **Predictability-conditioned optimization** | \(\mathrm{PI},\alpha,c^{\mathrm{mix}}\) 进入投影 | 外挂 “prediction module” |
| 3 | **Failure-safe adaptation** | Cor.4 优雅降级 | 启发式调参 / failure avoidance |

### 禁止定位

- ❌ “A prediction-enhanced projected gradient algorithm”  
- ❌ “SECDO outperforms all baselines on all metrics”  
- ❌ “achieves the theoretical regret bound”（对 Fig.4）  
- ❌ “guaranteed optimal”  

---

## 与数据的匹配

| 现象 | 叙事 |
|------|------|
| gap ↓、violation 略高于 reactive | 提前优化 ↔ 保守可行 的设计权衡 |
| Oracle 双指标最优 | 上界参照，不可部署 |
| A3 干净时遗憾≈Full，crash 崩 | \(\alpha\) 是 **failure containment**，非刷分旋钮 |
| Fig.4 拟合 \(a,b\) | **scaling validation**，非数值验证定理常数 |

---

## 章节入口

| 文档 | 作用 |
|------|------|
| `tables/evidence_chain.md` | Claim ↔ Theory ↔ Evidence |
| `reviewer_attack/R1_*.md` | 理论攻击预演 |
| `reviewer_attack/R2_*.md` | 算法攻击预演 |
| `reviewer_attack/R3_*.md` | 实验攻击预演 |
| `sections/experiments.md` | Fig.4/5/消融冻结措辞 |
