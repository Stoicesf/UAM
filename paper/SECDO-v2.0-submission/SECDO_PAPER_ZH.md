# 自演化约束分布式优化（SECDO）

**副标题：** 动态可行域下的预测性约束演化  
**版本：** SECDO-v2.0-submission（与英文投稿稿 `main.tex` / `main.pdf` 对齐的中文阅读版）  
**图件目录：** `figures/`（与英文稿相同）

> **一句话定位：** 面向*演化约束*的学习增强在线优化框架——预测质量决定何时提前优化有益，鲁棒机制防止预测失败时的性能崩塌。  
> **不是：** “更好的无人机优化器” / “预测增强的投影梯度模块” / 全指标 Pareto 最优。

---

## 摘要

本文提出 **SECDO**（Self-Evolving Constrained Distributed Optimization），一种面向**演化约束**的预测性优化框架。SECDO (i) 学习未来约束动力学，(ii) 执行可预测性条件化的提前 / 混合投影，(iii) 给出动态遗憾上界并在预测失败时优雅降级。本文不宣称全面优越，而是在所评估的动态约束场景中展示有利的最优性—安全性折中，并在预报不可靠时保持鲁棒。

**关键词：** 在线优化；动态约束；提前投影；动态遗憾；学习增强优化；无人机网络

---

## 一、引言

约束在线优化常把可行域 \(\mathcal{B}_t\) 视为已知、静态，或仅在决策之后揭示。在多无人机带宽分配等赛博物理系统中，绑定约束本身是动态的：标量容量
\[
c_t = W\log_2\bigl(1+\overline{\mathrm{SINR}}_t\bigr)
\]
诱导
\[
\mathcal{B}_t=\mathcal{B}(c_t)=\{x\ge 0:\mathbf{1}^\top x\le c_t\}.
\]
对当前 \(c_t\) 做反应式投影会滞后于集合运动；脆弱的“先预报再投影”管线则可能在预报失败时放大违规。

本文研究 **SECDO**：面向动态可行优化的**预测性约束演化**框架——而非“预测增强的投影梯度模块”。SECDO 识别约束演化、预估 \(\hat{\mathcal{B}}_{t+1}\)，并投影到可预测性条件化的混合预算上：
\[
\mathcal{B}_t
\;\rightarrow\;
\hat{\mathcal{B}}_{t+1}
\;\rightarrow\;
\Pi
\;\rightarrow\;
x_{t+1}.
\]

记
- \(\delta_t=d_H(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1})\)：**可行域失配**；
- \(\chi_t=d_H(\mathcal{B}_{t+1},\mathcal{B}_t)\)：**约束漂移**；
- \(\mathrm{PI}_t=\delta_t/(\chi_t+\varepsilon)\)：**可预测性指数**；
- \(\alpha_t=1/(1+\mathrm{PI}_t^2)\)：在提前预算与反应预算之间连续插值。

**贡献（仅三条）：**

1. 将演化约束表述为预测性优化问题，并把学到的集合失配与可行域不确定性联系起来（定理 1）。  
2. 提出带提前 / 混合投影的 SECDO，并建立由漂移与集合失配支配的动态遗憾**上界**（定理 2）。  
3. 通过可预测性（\(\delta_t<\chi_t\)）刻画何时提前投影收紧**可行性证书**，并给出失败安全降级（定理 3、推论 4）。

实证上，平均漂移增大约 \(13\times\) 时，累积动态遗憾增大约 \(56.7\times\)，违规率仍低于 \(2\%\)；崩溃注入与消融隔离了提前性与失败遏制的作用。

**图 1**（`figures/fig1_framework.pdf`）：状态编码 → 约束预测 → 可预测性条件化混合投影。

---

## 二、问题形式化

### 2.1 静态目标、动态约束

每一时刻 \(t\)：
\[
\min_{x\in\mathbb{R}^n} F_t(x)
\quad\text{s.t.}\quad
x\in\mathcal{B}_t=\mathcal{B}(c_t)=\{x\ge 0:\mathbf{1}^\top x\le c_t\}.
\]
记 \(x_t^\star\in\arg\min_{x\in\mathcal{B}_t}F_t(x)\)。  
**不声称**收敛到移动的无约束最优。

### 2.2 无人机容量教师

\[
c_t=W\log_2\bigl(1+\overline{\mathrm{SINR}}_t\bigr),\qquad
\chi_t=\lvert c_{t+1}-c_t\rvert=d_H(\mathcal{B}_{t+1},\mathcal{B}_t).
\]

### 2.3 学习到的约束动力学

\[
\hat c_{t+1}=\mathcal{F}_\phi(h_t),\qquad
\delta_t=d_H\bigl(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1}\bigr)
\le \lvert\hat c_{t+1}-c_{t+1}\rvert.
\]
SECDO 建立的是容量残差与可行域失配 \(\delta_t\) 之间的**可控关系**——**不声称**精确学会真实未来约束。

---

## 三、理论分析

理论链：
\[
\text{约束可识别}
\rightarrow
\text{投影扰动}
\rightarrow
\text{动态遗憾}
\rightarrow
\text{可预测性优势}
\rightarrow
\text{失败安全恢复}.
\]

### 符号冻结（必读）

| 符号 | 唯一含义 | 勿混称 |
|------|----------|--------|
| \(\delta_t\) | \(d_H(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1})\) 可行域失配 | 不要叫“预测误差”（容量残差写 \(\lvert\hat c-c\rvert\)） |
| \(\chi_t\) | \(d_H(\mathcal{B}_{t+1},\mathcal{B}_t)\) 约束漂移 | 不要叫“环境不确定性” |
| \(\epsilon_t\) | \(\|s_{t+1}-\hat s_{t+1}\|\) 状态预测误差 | 勿与 \(\delta\) 混用 |

### 3.1 约束演化可识别性

**定理 1（约束演化稳定性）。**  
对标量预算 \(\mathcal{B}(c)=\{x\ge 0:\mathbf{1}^\top x\le c\}\)，
\[
\delta_t
=
d_H\bigl(\mathcal{B}(c_{t+1}),\mathcal{B}(\hat c_{t+1})\bigr)
\le
\lvert c_{t+1}-\hat c_{t+1}\rvert.
\]
若 \(c=T(s)\) 为 \(L_g\)-Lipschitz 且 \(\hat c_{t+1}=T(\hat s_{t+1})\)，则 \(\delta_t\le L_g\epsilon_t\)。

含义：容量 / 状态残差 → 可行域失配的传播关系；**不是**预测器必然准确。

**引理（代理一致性 / Surrogate Consistency）。**  
设 \(L_c=\lvert\hat c-c\rvert^2\)，则
\[
\mathbb{E}[\delta_t]
\le
\mathbb{E}\lvert\hat c-c\rvert
\le
\sqrt{\mathbb{E}[L_c]}.
\]
因此最小化约束预测损失会收缩理论中的集合失配项；**不声称**最小化 \(L_c\) 即最小化 \(\mathrm{Reg}_T\)。

### 3.2 投影误差传播

**引理 2（投影扰动）。**  
在有界投影敏感性下，对任意 \(y\)，
\[
\bigl\|\Pi_{\hat{\mathcal{B}}}(y)-\Pi_{\mathcal{B}}(y)\bigr\|
\le
C_\Pi\, d_H(\hat{\mathcal{B}},\mathcal{B}).
\]
从而 \(\|x_{t+1}-x_{t+1}^{\circ}\|\le C_\Pi\delta_t\)。  
在 \(\mu\)-强凸下，\(\|x_{t+1}^\star-x_t^\star\|\le\mu^{-1}(\omega_t+L_g\chi_t)\)。

**注记：** \(C_\Pi\) 刻画投影算子敏感性，而非任意凸集的普适常数。该假设**不对任意凸集要求成立**，但对本文所考虑的多面体（标量预算）可行域成立；此时常有 \(C_\Pi=O(1)\)。

### 3.3 动态变分遗憾

定义
\[
\mathrm{Reg}_T=\sum_{t=0}^{T-1}\bigl(F_t(x_t)-F_t(x_t^\star)\bigr),\qquad
P_T=\sum_{t}\|x_{t+1}^\star-x_t^\star\|.
\]

**定理 2（动态遗憾上界）。**  
SECDO 具有由目标漂移、约束演化与可行域失配支配的动态遗憾上界：
\[
\mathrm{Reg}_T
\le
O\Bigl(\sqrt{T(1+P_T)}\Bigr)
+
O\Bigl(\sum_t(\epsilon_t+\delta_t)\Bigr),
\]
且
\[
P_T
\le
\frac1\mu\sum_t\bigl(\omega_t+L_g\chi_t\bigr).
\]

**不声称** SECDO 达到最优遗憾率。

### 3.4 可预测性条件化的可行性改进

反应 / 提前证书：\(V_R(t)\le\chi_t\)，\(V_A(t)\le\delta_t\)；\(\mathrm{PI}_t=\delta_t/(\chi_t+\varepsilon)\)。

**定理 3（Predictability-Conditioned Feasibility Improvement）。**  
若 \(\mathrm{PI}_t<1\)（即 \(\delta_t<\chi_t\)），则当集合失配小于约束漂移时，提前投影提供**更紧的约束违规证书**：
\[
\delta_t<\chi_t
\;\Rightarrow\;
\text{提前可行性证书严格紧于反应式}.
\]

这是关于**可行性证书**的陈述，不是目标函数优越性；**不是** “SECDO 总是优于反应式”。

### 3.5 失败安全恢复

**推论 4（优雅降级）。**  
取 \(\alpha_t=1/(1+\mathrm{PI}_t^2)\)，\(c^{\mathrm{mix}}_t=\alpha_t\hat c_{t+1}+(1-\alpha_t)c_t\)。  
当 \(\mathrm{PI}_t\gg 1\) 时 \(\alpha_t\to 0\)，SECDO **优雅退回**反应行为。记 \(I_t^{\mathrm{fail}}=\mathbf{1}\{\mathrm{PI}_t>1\}\)，
\[
\mathrm{Reg}_{\mathrm{SECDO}}
\le
\mathrm{Reg}_{\mathrm{Reactive}}
+
O\Bigl(\sum_t I_t^{\mathrm{fail}}\delta_t\Bigr).
\]

**不声称**完全恢复到无失败轨迹。

---

## 四、算法

### 算法 1：带预测自适应投影的 SECDO

（实现采用理论二值体制 \(\alpha\in\{0,1\}\) 的**连续松弛**。）

**输入：** 步长 \(\eta\)；预测器 \(\mathcal{F}_\phi\)；编码器 / GRU；目标 \(F\)。

对 \(t=0,\ldots,T-1\)：

1. \(z_t=\mathrm{Enc}(s_t)\)，\(h_t=\mathrm{GRU}(z_t,h_{t-1})\)  
2. \(\hat s_{t+1}=\mathrm{Dec}(h_t)\)，\(\hat c_{t+1}=\mathcal{F}_\phi(h_t)\)  
3. \(y_t=x_t-\eta\nabla F(x_t)\)  
4. 估计 \(\hat\delta_t,\hat\chi_t\)；\(\widehat{\mathrm{PI}}_t=\hat\delta_t/(\hat\chi_t+\varepsilon)\)  
5. \(\alpha_t=1/(1+\widehat{\mathrm{PI}}_t^2)\)  
6. \(c^{\mathrm{mix}}_t=\alpha_t\hat c_{t+1}+(1-\alpha_t)c_t\)  
7. \(x_{t+1}=\Pi_{\mathcal{B}(c^{\mathrm{mix}}_t)}(y_t)\)　← **单次投影**  
8. 观测 \(s_{t+1},c_{t+1}\)；可选在线小步更新 \(\phi\)

**理论 vs 实现：**  
定理 2–3 比较的是投影到 \(\hat{\mathcal{B}}_{t+1}\) 与 \(\mathcal{B}_t\) 的极端（对应 \(\alpha=1\) 与 \(0\)）。部署算法用预算混合 \(c^{\mathrm{mix}}\)（\(0<\alpha_t\le 1\)）连续插值，并保持**一次**投影，避免两投影凸组合带来的额外开销。自适应 \(\alpha\) 是实现层鲁棒机制。

**训练 ≠ 在线算法。**  
离线最小化 \(L=0.5L_s+1.0L_c+0.1L_u\)。最小化 \(L_c\) 收缩 \(\delta\)，但**不直接**最小化 \(\mathrm{Reg}_T\)。训练时 \(\hat c\) 对 \(\Pi\) 停止梯度。

---

## 五、实验

SECDO **不以**全指标 Pareto 最优为目标。三类实验：

1. **合成凸问题：** 验证定理 2 的**尺度结构**（非刷榜）。  
2. **无人机系统评估：** 展示实用收益与安全/效率折中（**不是**“验证定理”）。  
3. **预测失败压力测试：** 对应推论 4。

### 5.1 动态遗憾尺度（图 4）

**图：** `figures/fig4_regret.pdf`  

经验 \(\mathrm{Reg}_T\) 跟随结构 \(O\sqrt{T(1+P_T)}+O\sum\delta\) 增长。图中拟合包络**仅用于可视化**，不代表普适理论常数。

### 5.2 UAV 漂移敏感性（表 I）

| 工况 | \(\bar\chi\) | \(\mathrm{Reg}_T\) | 违规率 |
|------|-------------:|-------------------:|-------:|
| Slow | 0.010 | \(7.69\times10^{-4}\) | 0.33% |
| Fast | 0.131（×13.0） | \(4.36\times10^{-2}\)（×56.7） | 1.46% |
| Stress | 0.203 | \(6.86\times10^{-2}\) | 1.89% |

\(\bar\chi\) 增大约 \(13\times\) → \(\mathrm{Reg}_T\) 增大约 \(56.7\times\)，违规仍有界。

### 5.3 失败恢复与 PI 自适应（图 2、图 3）

- **图 2** `figures/fig2_crash.pdf`：预测器崩溃协议下，相对 \(\alpha\equiv 1\) 的优雅降级。  
- **图 3** `figures/fig3_PI_boundary.pdf`：腐败窗内 \(\mathrm{PI}_t\) 与 \(\alpha_t\) 的反向关系。

### 5.4 最优性—安全性折中（图 5）

**图：** `figures/fig5_uav.pdf`  
**标题含义：** 动态约束下的最优性—安全性折中。  

SECDO 相对保守反应基线取得**有利的 gap–violation 折中**（间隙更低、违规略高）。Oracle 使用未来 \(c_{t+1}\)，仅为信息上界，不可部署。

### 5.5 消融（表 II）

| 变体 | Fast \(\mathrm{Reg}_T\) | Crash 窗内上升 |
|------|------------------------:|---------------:|
| Full SECDO | 0.044 | 0.185 |
| A1/A2（无提前） | 0.052（+18%） | 0.142 |
| A3（\(\alpha\equiv 1\)） | 0.043 | 0.559（×3） |

去掉提前性使 Fast 遗憾约升 18%。  
自适应 \(\alpha\) **主要提升鲁棒性而非名义性能**：预报准确时与 \(\alpha\equiv 1\) 接近；失败时遏制误差放大（推论 4）。

---

## 六、结论

SECDO 是面向动态可行优化的预测性约束演化框架：学习 \(\hat{\mathcal{B}}_{t+1}\)，施加可预测性条件化混合投影，并给出动态遗憾保证与预测失败下的优雅降级。理论—算法—实验构成一条链：

\[
\text{约束演化}
\rightarrow
\text{提前/混合投影}
\rightarrow
\text{自适应鲁棒}
\rightarrow
\text{遗憾分析},
\]

而非一堆互不相关的启发式堆叠。

**局限：** 当前假定演化可行域具有有界投影敏感性，并聚焦于多面体（标量预算）几何下的可预测约束演化。向强非凸流形、更强部分可观测、以及完整拓扑耦合经典求解器的扩展，是有意义的未来方向。UAV 上的 DSGF/AC-DSGF 基线是在共享容量教师下的近视适配器，不应解读为对所有历史变体的穷尽复现。

---

## 附录提要

### A. 计算复杂度

| 方法 | 预测 | 投影 | 更新 |
|------|------|------|------|
| Reactive | — | \(O(P)\) | \(O(d)\) |
| Oracle | 已知未来 \(c_{t+1}\) | \(O(P)\) | \(O(d)\) |
| SECDO | \(O(C_\phi)\) 推理 | \(O(P)\)（**单次**） | \(O(d)\) |

相对反应式，SECDO 增加预测推理 \(O(C_\phi)\) 与标量 PI/α 运算，**不增加投影次数**。

### B. 证据链（简表）

| 论断 | 理论 | 证据 |
|------|------|------|
| 集合失配可控 | 定理 1 | \(\delta\) 日志 / \(L_c\) |
| 投影扰动有界 | 引理 2 | Oracle 间隙 |
| 遗憾对漂移敏感 | 定理 2 | 图 4；表 I |
| \(\delta<\chi\) 时证书更紧 | 定理 3 | 图 3 |
| 失败优雅降级 | 推论 4 | 图 2；消融 A3 |

---

## 阅读指引

| 你想看 | 打开 |
|--------|------|
| 英文投稿 PDF | [`main.pdf`](main.pdf) |
| 本中文通读版 | 本文 `SECDO_PAPER_ZH.md` |
| 图 | [`figures/`](figures/) |
| 封面信（英） | [`COVER_LETTER.md`](COVER_LETTER.md) |
| 投稿检查 | [`PHASE6_SUBMISSION_RELEASE.md`](PHASE6_SUBMISSION_RELEASE.md) |

---

*中文版仅供阅读与内部讨论；对外投稿以英文 `main.tex` / `main.pdf` 为准。*
