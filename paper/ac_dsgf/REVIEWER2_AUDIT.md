# AC-DSGF v1 — Final Consistency Audit + Reviewer #2 Risk Card

**约束最高优先级：** [`LANGUAGE_FREEZE.md`](LANGUAGE_FREEZE.md)  
**主文：** `AC_DSGF_CN.md` / `AC_DSGF_EN.md`  
**日期：** 2026-07-18

---

## A. Consistency Audit（已执行）

| 检查项 | 状态 | 备注 |
|--------|------|------|
| Abstract 问题句 | ✅ | predefined structures / limited adaptability |
| Abstract 方法句 | ✅ | constrained optimization + AC-DSGF task-aware selection |
| Abstract 结果句 | ✅ | identical budgets + competitive performance + scalability |
| Gap1 | ✅ | lacks adaptability（非 high cost） |
| Gap2 | ✅ | attention ≠ topology（固定英文句） |
| Gap3 | ✅ | jointly optimized with task objectives |
| Method 词汇 | ✅ | selection / sparse construction（非 pruning/reduction 主叙事） |
| Algorithm 1 名 | ✅ | Constraint-aware Adaptive Topology Selection |
| Prop.1 | ✅ | Communication Scalability Bound + 非 convergence 句 |
| Exp 结构 | ✅ | Setup / Adaptive selection / Scalability / Robustness / Fault-aware |
| 工程隔离 | ✅ | hardware-compatible framework；无 SwarmOS contribution |
| ++ 残留（主文） | ✅ | 仅出现在禁止声明中 |

---

## B. Reviewer #2 风险卡（避免失分）

### R2-1 创新性
**可能问：** 与 attention / Top-K 剪枝有何本质区别？  
**答：** Attention = feature weighting；AC-DSGF = **budget-constrained topology decision variable**（Algorithm 1 Stage1–3）。同预算下比较边质量，不是剪枝模块。

### R2-2 理论有效性
**可能问：** Prop.1 是否过声称？  
**答：** 仅 **scalability bound**（\(\eta_N\to0\)），明确 *not optimization/learning convergence*。任务保持为 Lipschitz 条件分析。

### R2-3 实验公平性
**可能问：** Soft Mass 更低是否因为预算更紧？  
**答：** 主结论使用 **identical communication budgets**；相对密图全开的 Soft Mass 仅为辅助代理。RULE/AC 边数对齐时不宣称 overhead reduction。

### R2-4 Claim 过强
**禁止再写：** significantly reduces communication；AI controller；AC-DSGF++；SwarmOS 算法贡献。  
**允许：** competitive under identical budgets；scalable；robust；task-aware selection。

---

## C. 创新边界（审稿人应看到的故事）

```
Problem: topology usually predefined
Insight: topology should be optimized
Method:  AC-DSGF constrained task-aware selection
Theory:  sparse topology → scalability
Experiment: identical budgets → effectiveness + robustness
```

**下一动作：** 仅语言级修补（见 [`AC_DSGF_v1_submission_checklist.md`](AC_DSGF_v1_submission_checklist.md)）；不扩算法、不堆实验、不把 SwarmOS 写入贡献。
