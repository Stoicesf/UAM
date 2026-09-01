# R3 — Experiment / Systems Reviewer Attack Simulation

**角色：** 实验与系统审稿人  
**目标：** 把 violation 权衡、Oracle gap、拟合界变成**主动叙事**  
**数据锚：** Fig.4/5、Ablation、`EVIDENCE_CHAIN.json`

---

## C1. “SECDO violation 比 Reactive 高，算法失败？”

**攻击：** Fig.5 左栏 SECDO viol 偏高。

**主动解释（冻结句）：**

> SECDO achieves a lower optimality gap by exploiting future constraint evolution, while maintaining **bounded** constraint violation through adaptive projection. The slightly increased violation compared with purely reactive methods reflects the inherent **optimality–safety trade-off** between anticipatory optimization and conservative feasibility.

**图题：** *Optimality–Safety Trade-off under Dynamic Constraints*（勿用 Performance comparison）。

---

## C2. “为何不全面超过所有方法？”

**回复：** 本文**不追求** Pareto 最优。定位是预测收益、有界安全与理论可解释性的平衡。Oracle 双优是信息上界；Reactive 更保守可行；SECDO 取中间机制并由 Cor.4 保底。

---

## C3. “Fig.4 是拟合界，不算验证定理？”

**回复：** 见 R1-A3。论文写 **scaling behavior / structure validation**，系数 fitted for visualization only。UAV Table I（\(\chi\times13\Rightarrow\mathrm{Reg}\times56.7\)）提供系统侧漂移敏感性证据，与 Fig.4 互补。

---

## C4. “Oracle gap 说明预测没用？”

**回复：** Oracle 用未来真 \(c_{t+1}\)；SECDO 不用。gap 相对 Reactive/DSGF 下降（Fig.5 右）说明**可部署预测**有收益；与 Oracle 的剩余差距是可学习性边界，不是负结果。

---

## C5. “Ablation A3 在干净 UAV 上遗憾更好，为何还要自适应 \(\alpha\)？”

**冻结解释：**

> The adaptive coefficient does **not** aim to improve nominal performance when predictions are accurate. Instead, it prevents error amplification under prediction failures, consistent with Corollary 4.

数字：干净 Fast 上 A3 \(\mathrm{Reg}_T\approx0.043\) vs Full \(0.044\)；crash 窗内上升 \(0.559\) vs \(0.185\)（×3）。

---

## C6. “A1/A2 同数，消融不独立？”

**回复：** 在标量预算实现中 \(\alpha=0\) 与无 \(\hat c\) 的反应投影决策重合，属机制等价，不是实验错误。报告两者以对应 “去预测” 与 “去提前投影” 的概念消融；crash-synth 无学习 \(\hat c\) 时 A2≡A1 已注明。

---

## C7. “Stress 1.89% 可接受吗？种子是否够？”

**回复：** 相对 Slow 上升但仍有界、无崩溃；与 Cor.4 一致。主文报 5 seeds 均值；附录给全表。不设单一绝对安全阈值，强调相对基线与有界性。

---

## C8. “有没有 energy / throughput？”

**回复：** 当前记录 gap（效用代理）与 violation；**不编造**未记录的 energy/throughput。若审稿要求，属 future measurement，非现有 claim。

---

## R3 检查清单

- [ ] Fig.5 标题含 trade-off  
- [ ] 正文主动解释 viol↑  
- [ ] A3 叙事 = failure containment  
- [ ] Fig.4 = scaling only  
- [ ] Evidence chain 表可引用  
