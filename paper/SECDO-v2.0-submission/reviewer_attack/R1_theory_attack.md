# R1 — Theory Reviewer Attack Simulation

**角色：** 优化 / 学习理论审稿人  
**目标：** 提前堵死假设、常数与 claim hygiene 攻击  
**叙事红线：** [`../NARRATIVE_POSITIONING.md`](../NARRATIVE_POSITIONING.md)

---

## A1. “你们假设太强：强凸 + 投影 Lipschitz？”

**攻击：** Assump.2 \(\mu\)-强凸、Assump.3 \(C_\Pi\) 不现实。

**回复要点：**

1. 强凸用于控制 \(\|x_{t+1}^\star-x_t^\star\|\) 与 \(P_T\)；UAV 二次分配代理满足。  
2. 若仅凸不强凸，主项可改为标准 online convex 形式，漂移项仍在；文中标明标量预算特例下 \(C_\Pi=O(1)\)。  
3. **不声称**一般非凸约束成立。

**指向：** `appendix/A_assumptions.md` Assump.2–3；`B_proofs.md` B.2。

---

## A2. “Projection sensitivity \(C_\Pi d_H\) 是手波？”

**攻击：** 为何不是 \(\|\Pi_A-\Pi_B\|\le d_H\)？

**回复：** 正确点态形式是
\[
\|\Pi_A(y)-\Pi_B(y)\|\le C_\Pi d_H(A,B),
\]
紧凸多面体上 \(C_\Pi<\infty\) 为标准；标量预算可由 Thm.1 几何取 \(C_\Pi=O(1)\)。我们**禁止**无 \(C_\Pi\) 的算子写法。

---

## A3. “Theorem 2 的常数在哪里？Fig.4 是事后拟合？”

**攻击：** \(a,b\) 拟合 ⇒ 你在 data-fit 定理。

**回复（必须主动承认）：**

- Thm.2 给出的是 **\(O(\cdot)\) 结构**，不是可即插即用的数值常数。  
- Fig.4 仅做 **scaling validation**：经验 \(\mathrm{Reg}_T\) 随 \(\sqrt{T(1+P_T)}\) 形态增长，并落在同结构上包络内。  
- \(a,b\) **仅用于可视化**，不声称普适理论常数（见 `fig4_theory_bound_meta.json` note）。

**论文句：** 见 `sections/experiments.md` §Fig.4 冻结措辞。

---

## A4. “动态遗憾定义是否标准？\(F_t\) vs 静态 \(F\)？”

**攻击：** UAV 用静态代理，定理写 \(F_t\)。

**回复：** 静态 \(F\) 是 \(F_t\equiv F\)、\(\omega_t=0\) 特例；界仍含 \(\sum\chi_t\) 经 \(P_T\)。主文已声明 claim = prediction-aware gap / dynamic feasible variational regret，**不是**无约束移动最优收敛。

---

## A5. “Theorem 3 说 always better？”

**攻击：** anticipatory 总优于 reactive？

**回复：** **否。** 仅当 \(\delta<\chi\)（\(\mathrm{PI}<1\)）时 **证书上界**更紧；否则 Cor.4 降权。措辞是 *tighter upper bound*，不是路径上点态 \(V_A<V_R\) 恒成立。

---

## A6. “Corollary 4 的 \(O(\sum I^{\mathrm{fail}}\delta)\) 是否松？”

**攻击：** 恢复界太松 / 启发式 \(\alpha=1/(1+\mathrm{PI}^2)\)。

**回复：**

1. 机制目标是 **failure containment**（混合预算连续退回 \(c_t\)），不是最优恢复速率。  
2. \(\alpha\) 形式保证 \(\mathrm{PI}\to\infty\Rightarrow\alpha\to0\)，且对 \(\mathrm{PI}\) 光滑。  
3. Ablation A3：干净工况几乎不损；crash 窗内上升 ×3 → 实证支撑“遏制”而非“刷分”。

---

## A7. “Hausdorff / 标量预算太特殊，推不到一般 \(g(x,s)\)？”

**回复：** 主定理在标量预算上写死几何；Assump.1 给出一般 Lipschitz 链接 \(\delta\le L_g\epsilon\) 作为扩展接口。UAV Shannon 教师是一阶应用，不声称任意非凸 \(g\)。

---

## R1 检查清单

- [ ] 主文无 “achieves the bound”  
- [ ] Fig.4 caption 含 fitted / visualization only  
- [ ] Thm.3 含 conditional  
- [ ] Cor.4 含 graceful degradation  
