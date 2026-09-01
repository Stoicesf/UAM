# 面向通信约束多无人机蜂群协同的自适应拓扑学习

**方法：** **AC-DSGF**（Adaptive Constraint-aware Dynamic Sparse Graph Framework），**v1 冻结**

> 仅 Markdown 稿件。主文禁止出现：AC-DSGF++、World Model、SwarmOS、PX4、Runtime/Adapter 等工程栈专名。英文主文见 `../ac_dsgf/AC_DSGF_EN.md`。

---

## 摘要

现有多无人机蜂群策略常将通信拓扑视为固定基础设施（全连接、半径图或 \(k\) 近邻图）。注意力权重改进了消息聚合，但在带宽限制下并不决定链路是否应被激活——消息仍可能在加权之前就被传输。我们在软预算约束下研究无人机应*何时、与谁*通信，并提出 **AC-DSGF**，通过 Candidate Edge Scoring、Budget-Constrained Edge Selection 与 Residual Recovery 学习自适应通信稀疏模式。度约束下的规模界给出 \(\eta_N=\mathcal{O}(1/N)\)。在相同通信预算下，AC-DSGF 取得具有竞争力的协同性能，同时在不同蜂群规模上保持显著更低的 Soft Communication Activation（SCA）。

**关键词：** 自适应通信拓扑；软预算约束；无人机蜂群协同；通信激活；多智能体强化学习

---

## 1 引言

### 1.1 Layer 1 — 拓扑被当作固定基础设施

多无人机蜂群通常依赖**预定义**通信结构——全连接、半径图或 \(k\) 近邻——其中
\[
A_t=f(x_t)
\]
由几何关系固定。在此视角下，拓扑是*基础设施*，而非决策变量。

### 1.2 Layer 2 — Attention ≠ 通信激活

许多 MARL 方法学习注意力权重 \(\alpha_{ij}\) 用于**消息聚合**。聚合加权回答的是融合时*谁更重要*，但通常默认消息已经可用。在带宽限制下，运维问题是链路是否应被**激活**（*谁应通信*）。

### 1.3 Layer 3 — 部署中的软预算

带宽、能耗与干扰迫使采用软通信预算。因此我们追问：

> **无人机应何时、与谁通信？**

![Fig.0 动机](../ac_dsgf/figures/Fig0_motivation_topology.png)

**Fig. 0**（a）作为基础设施的固定几何拓扑；（b）attention 融合（重要性 ≠ 激活）；（c）软预算下所学自适应稀疏模式 \(G_t\)。

### 1.4 方法与贡献

AC-DSGF 在软预算下通过 Candidate Edge Scoring、Budget-Constrained Edge Selection 与 Residual Recovery 学习自适应通信稀疏，训练代理目标为 \(\mathbb{E}[R-\lambda_c C]\)。

1. **问题：** 将自适应通信拓扑学习表述为蜂群任务性能与通信代价的联合优化问题。  
2. **方法：** 提出带预算正则与残差引导的可微拓扑适应机制，联合考虑任务相关性与通信预算（Candidate Edge Scoring → Budget-Constrained Edge Selection → Residual Recovery）。
3. **验证：** 验证在不同蜂群规模与预算化部署下，可比协同性能的同时显著降低通信激活（SCA）。

---

## 2 相关工作

### 2.1 多智能体系统中的通信学习

CommNet、DIAL、TarMAC、ATOC、IC3Net 等聚焦**消息**（内容、门控、注意力）。支撑图多为全连接或固定邻域；显式带宽预算与可部署离散拓扑往往处于次要地位。**Message learning ≠ topology optimization.**

### 2.2 面向 UAV / 蜂群的图神经网络 MARL

GAT、DGN 与 Graph MARL 在关系结构上聚合，但 \(G\) 通常为**默认**半径/KNN 图。学习发生在给定 \(G\) 上，而非对 \(G\) 本身的学习。

### 2.3 通信高效 MARL

事件触发与事后稀疏化方法常在固定图上稀疏化，与任务耦合较弱。AC-DSGF 强调 **joint budget-constrained topology selection**（非 communication-pruning 叙事）。

---

## 3 问题表述：约束拓扑优化

### 3.1 任务与动态通信图

局部观测 \(o_i^{t}=[p_i^{t},v_i^{t},g_i^{t},\ell_i^{t}]\)。时变图 \(G_t=(V,E_t)\)，

\[
E_t=\{(i,j):g_{ij}^{t}>0\},\qquad g_{ij}^{t}\in[0,1],
\]

限制在通信半径内。Success \(S\) 为评估的目标到达比例。

### 3.2 约束优化

\[
C(G_t)=\sum_{i\neq j}g_{ij}^{t},\qquad C=\tfrac1T\sum_t C(G_t).
\tag{1}
\]

\[
\begin{aligned}
\max_{\pi,\{G_t\}}&\quad\mathbb{E}[R(\pi,\{G_t\})]\\
\mathrm{s.t.}&\quad C(G_t)\le B\quad(\text{或度预算 }K).
\end{aligned}
\tag{2}
\]

可微松弛：

\[
\max_{\theta}\;\mathbb{E}\Bigl[\sum_t(r_t-\lambda_c C(G_t))\Bigr].
\tag{3}
\]

我们研究**预算下的任务有效稀疏动态图**，而非通信压缩器。

---

## 4 方法：自适应通信拓扑学习（AC-DSGF）

![Fig.1 框架](figures/Fig1_framework.png)

**Fig. 1** 观测 → 拓扑学习器 → 自适应图 → 残差策略。

### 4.1—4.3 三阶段（AC-DSGF v1）

**Stage 1 — Candidate Edge Scoring.**  
\[
s_{ij}^{t}=f_\theta(z_i^{t},z_j^{t},m_t),\qquad g_{ij}^{t}=\sigma(s_{ij}^{t})\cdot A_{ij}^{t}.
\]
估计信息交换的潜在效用（非 attention 机制）。

**Stage 2 — Budget-Constrained Edge Selection.**  
\[
\max_{E_t}\sum_{(i,j)\in E_t}s_{ij}^{t}
\quad\mathrm{s.t.}\quad
|E_t|\le B
\quad\text{或}\quad C(G_t)\le B_c.
\]
目标 = 任务感知选择；约束 = 通信预算  
（*非* “maximize performance while minimizing communication”）。

当 \(|E_{\mathrm{AC}}|=|E_{\mathrm{RULE}}|\) 时，在 identical-budget 框架下比较边质量。

**Stage 3 — Residual Recovery.**  
\[
G_t=G_t^{\mathrm{opt}}\cup G_t^{\mathrm{res}},\qquad a=\pi(o)+\beta\Delta(\Phi).
\]
条件有界动作偏差 \(\|a-a^\star\|\le\beta\varepsilon\)（非最优性/收敛保证）。

### 4.4 Topology vs Attention

| | Attention \(\alpha_{ij}\) | Topology gate \(g_{ij}\) |
|--|---------------------------|---------------------------|
| 作用 | 融合 | 传输 / 边选择 |
| 对象 | 特征 | 网络 |
| 约束 | 无 | 带宽 / \(K\) |
| 部署 | 连续 | 离散发送集 |

### 4.5 Soft Communication Activation (SCA)

定义 **Soft Communication Activation (SCA)**：
\[
C_s=\sum_{i,j}g_{ij}
\]
作为训练/分析用的连续激活强度代理。  
SCA **不是**射频包计数。离散部署使用硬触发
\(C_{\mathrm{hard}}=\sum_{ij}\mathbf{1}(g_{ij}>\tau)\) 或 per-agent Top-\(K\)。

### 4.6 理论分析

#### Prop.1 — Communication scaling under degree constraint

*本命题刻画有界邻域假设下的通信密度规模关系；并非优化或学习收敛性声明。*

假设每个智能体最多与 \(K\) 个邻居通信，
\[
|E_t(i)|\le K
\quad\Rightarrow\quad
|E_t|\le NK.
\]
完全有向图满足 \(|E_{\mathrm{full}}|=N(N-1)\)。因此归一化密度满足
\[
\eta_N
=
\frac{|E_t|}{|E_{\mathrm{full}}|}
\le
\frac{K}{N-1}
=
\mathcal{O}\!\left(\frac1N\right).
\]
**解释。** 在有界邻域下，通信密度随蜂群规模呈反比阶下降。我们**不**声称实现边数精确等于 \(KN\)。

#### Prop.2 — Residual Stability Analysis

设 \(a^\star=\pi(o,m^\star)\) 为充分通信参考动作，\(a=\pi(o)+\beta\Delta(\Phi)\) 为稀疏消息下的残差修正动作。定义消息误差 \(\varepsilon=\|m^\star-m\|\)。若策略对消息参数 Lipschitz 连续，常数为 \(L_\pi\)，则
\[
\|a-a^\star\|
\le
\beta L_\pi\varepsilon.
\]
**解释。** Residual Recovery 抑制拓扑稀疏化引起的**动作突变**。这是动作的条件稳定性上界——**不是**任务回报 / Success 非递减的保证。


**推论 1（任务回报退化界）。** 在命题 2 的 Lipschitz 条件下，设单步奖励 \(r_t\) 对动作 \(a_t\) 为 \(L_r\)-Lipschitz。记 \(J_{\mathrm{full}}=\mathbb{E}[\sum_t r_t(a_t^\star)]\) 为充分通信下的期望回报，\(J_{\mathrm{sparse}}\) 为稀疏拓扑+残差修正下的回报，则：
\[
|J_{\mathrm{full}} - J_{\mathrm{sparse}}|
\;\le\;
L_r \cdot \beta L_\pi \cdot \varepsilon \cdot T,
\]
其中 \(\varepsilon = \max_t \|m_t^\star - m_t\|\) 为时域内最大消息偏差，\(T\) 为 episode 长度。

*解释.* 该推论将稳定性分析从*动作空间偏差*提升至*累积任务回报*。它给出性能差距被消息近似误差 \(\varepsilon\) 与残差增益 \(\beta\) 线性上界的条件，**不**声称稀疏化提升回报；而是给出 SCA 大幅降低时不至于灾难性性能退化的定量条件——与表 2 及 §5.5 平缓退化一致。


#### Prop.3 — Interpretation of Adaptive Topology Learning

训练最小化软预算代理目标 \(\mathbb{E}[R-\lambda_c C]\)，其中可微门控 \(g_{ij}=f_\theta(h_i,h_j,\cdot)\)。我们**不**声称对
\[
\max_{g}\sum_{ij}g_{ij}u_{ij}
\quad\mathrm{s.t.}\quad
\sum_{ij}g_{ij}\le B
\]
给出显式组合优化求解器。相反，所学门控机制可**解释为**预算约束边选择问题的**近似/诱导**解：Stage 2 按所学分数排序候选边并施加预算选择，而非固定图上的事后幅度剪枝。


**推论 2（有界效用估计下的 Top-K 选择）。** 设激活边 \((i,j)\) 的真实边际效用为 \(u_{ij}\in[0,1]\)，所学分数为 \(\hat{s}_{ij}\)，且估计误差 \(\|\hat{s}_{ij} - u_{ij}\|_\infty \le \delta\)。对每智能体预算 \(K\)，记 \(E_{\mathrm{opt}}\) 为最大化 \(\sum u_{ij}\) 且满足 \(|E_i|\le K\) 的最优边集，\(E_{\mathrm{top}}\) 为按 \(\hat{s}_{ij}\) 排序选择的边集。则效用差距满足：
\[
\sum_{E_{\mathrm{opt}}} u_{ij}
-
\sum_{E_{\mathrm{top}}} u_{ij}
\;\le\;
2K\delta.
\]

*解释.* 该推论为 Fig. 9 的 Top-\(K\) 部署提供理论支撑：只要所学分数在误差 \(\delta\) 内有界，按 \(\hat{s}\) 排序得到的硬拓扑效用与最优 \(K\) 近邻选择差距为 \(\mathcal{O}(K\delta)\)。**不**声称 Top-\(K\) 训练优于软训练；而是解释为什么软学分数在推理阶段可离散化为有竞争力的硬链路。


### 4.7 与事后稀疏化 / 随机删边对比

| | Random drop / post-hoc sparsification | Attention-only | AC-DSGF |
|--|---------------------------------------|----------------|---------|
| 形式 | 固定图上删边 | 固定图上加权 | 约束拓扑选择 |
| 改变拓扑 | 启发式 | 否 | 是（边选择） |
| 任务耦合 | 弱 | 间接 | \(\max R\) s.t. \(C\le B\) |

**Algorithm 1: Adaptive Communication Topology Learning**

```
Input:  node embeddings h_i, mission context m_t, budget B (or K)
1  Candidate Edge Scoring          s_ij = f_theta(z_i, z_j, m_t)
2  Differentiable relaxation       g_ij = sigma(s_ij) * A_ij
3  Budget-Constrained Selection    E_t <- Top-K / budget on g  (|E_i|<=K)
4  Message aggregation             m_i <- Aggregate(G_t)
5  Residual action correction      a <- pi(o) + beta Delta(Phi)
Output: adaptive graph G_t and action a
```

预算进入学习回路（步骤 2—3），而非仅作事后剪枝。

![Fig.2 流水线](figures/Fig3_algorithm_flow.png)

**Fig. 2** 约束自适应拓扑学习流水线。

---

## 5 实验

### 5.1 实验设置

VMAS navigation；\(N{=}16\)；\(R_c{=}0.5\)；MAPPO；评估使用 TorchRL，在 `env.step` 后调用 `step_mdp`。基线：GAT、DSGF、AC-DSGF。

### 5.2 主结果

**Table 1** 默认评估协议下的主对比（\(N{=}16\)，预算比 \(\rho{=}1\)）。  
Success 为 episode 均值完成率（%）。SCA 为 Soft Communication Activation \(C_s=\sum g_{ij}\)（训练代理，非包计数）。

| Method | Success (%) | SCA \(C_s\) | CEI |
|--------|------------:|-------------:|----:|
| GAT | 21.3 | 44.14 | 0.0048 |
| DSGF | 21.7 | 40.29 | 0.0054 |
| **AC-DSGF** | **27.1** | **0.0081** | **33.5** |

> **口径说明。** Table 1 为主方法对比。Table 2 在冻结 AC checkpoint 上进行 *eval-only* 门控干预，**数值不可与 Table 1 直接互换**。

主主张：**相同通信预算下的任务感知选择**。相对稠密全开基线，SCA 仅为激活强度代理。图注强调 *Topology Selection / Adaptation under Communication Constraints*。

![Fig.3 Pareto](figures/Fig4_pareto.png)

**Fig. 3** Success—SCA 权衡（\(x\) 轴对数尺度）。

### 5.3 稠密通信分析

![Fig.4 预算](figures/Fig7_comm_density.png)

**Fig. 4** 预算比 vs Success。额外边带来的边际任务收益有限（饱和）。强制 \(g{=}A\) 使 SCA 升至约 40.6，而 Success 仅从 24.6% 升至 28.9%（Table 2 协议）。

### 5.4 消融与 Random-Drop 基线

**Table 2** 在冻结 AC-DSGF checkpoint 上的固定通信干预（eval-only；表内 episode 预算一致）。要点：AC-random Success 略高，但 SCA 约为 AC-full 的 **~2,600×**（20.32 vs 0.0077）。

| Variant | Gate | Budget | Residual | Success (%) | SCA \(C_s\) |
|---------|:----:|:------:|:--------:|------------:|-------------:|
| DSGF | × | × | ✓ | 21.7 | 40.29 |
| AC-random (Random Drop) | random | × | ✓ | 28.5 | 20.32 |
| AC-no budget (\(g{=}A\)) | open | × | ✓ | 28.9 | 40.63 |
| **AC-full** | ✓ | ✓ | ✓ | 24.6 | **0.0077** |
| w/o Residual (\(N{=}4\)) | — | — | × | 0.22 | — |
| Full DSGF (\(N{=}4\)) | — | — | ✓ | 9.27 | — |


*残差消融是在较小 \(N{=}4\) 设定下的**机制验证**，用以隔离静默坍塌（9.27%→0.22%）；\(N{=}16\) 主结果使用同一残差模块，不宜过度外推为 \(N{=}16\) 的 Success 主张。*

**关于 AC-random ≥ AC-full Success。** 随机稀疏化可能因噪声抑制而偶然提升任务表现，但**不提供等价预算下的通信效率保证**。

> **尽管 AC-random 的 Success 略高于 AC-full，其 SCA 约为后者的 2,600 倍（20.32 vs 0.0077），在数量级上违反通信预算。在硬预算约束下（例如每智能体 ≤\(K\) 条边），AC-random 无法保证可行调度；而 AC-full 的稀疏拓扑可直接部署。**

全开边（\(g{=}A\)）恢复稠密强度，但无法复现 AC-full 的高 CEI 工作区。收益来自**联合学习的选择性拓扑**，而非任意随机稀疏化。

AC-DSGF 的优势不在于不计代价地最大化 Success，而在于以**近零激活密度取得可比或略低的 Success**——这是扩展到带宽受限无人机蜂群的前提。

### 5.4b 固定 Hard Top-\(K\) 预算

为在*相同硬预算*下隔离选择质量，我们冻结 AC actor，以 per-agent Top-\(K{=}2\) 执行三种排序规则：Random、Distance（\(-\mathrm{dist}\)）与所学 AC 分数。硬边数由构造匹配（\(\approx NK\)）。

**Table 2b** 固定硬预算（\(K{=}2\)，\(N{=}16\)，64 episodes；Success 为 mean\(\pm\)std %）

| Method | Success (%) | Hard edges \(\|E\|\) |
|--------|------------:|---------------------:|
| Random Top-\(K\) | \(25.1\pm13.0\) | \(26.7\pm2.0\) |
| Distance Top-\(K\) | \(29.1\pm12.7\) | \(26.6\pm1.5\) |
| AC Top-\(K\) | \(28.2\pm11.7\) | \(26.3\pm2.0\) |

在严格 hard Top-\(K\) 部署下，AC-DSGF 与启发式策略取得**可比**性能（Distance 略高；二者均高于 Random），表明所学拓扑**并不**依赖过量通信冗余。AC-DSGF 的主要优势并非“硬掩码下的最佳 \(K\) 邻居”，而是在软预算下学习**自适应软稀疏模式**（Table 1 / Fig. 11）：SCA 比稠密基线低数个数量级，同时协同仍具竞争力。Fig. 10 进一步表明边重要性与任务对齐，超出均匀稀疏化。

![Fig.5 Hard Top-K](../ac_dsgf/figures/Fig14_hard_topk.png)

**Fig. 5** 固定硬预算对比（per-agent Top-\(K{=}2\)）。

### 5.5 预算退化与失败边界

| \(\rho\) | GAT | DSGF | AC-DSGF |
|---------:|----:|-----:|--------:|
| 100% | 21.3 | 21.7 | 27.1 |
| 50% | 18.2 | 21.6 | 25.2 |
| 10% | 16.9 | 21.7 | 24.2 |


> **关于 DSGF 在预算扫描下的平坦曲线。** DSGF 无学习门控，有效边由固定半径决定。对其施加事后 Top-\(K\)/预算掩码时 Success 变化很小，因为在本稠密蜂群设定下，其有效邻域规模相对激进的 10% 预算削减已处于饱和区。平坦曲线反映的是 **DSGF 的预算饱和区**，而非 DSGF 未遵守评估掩码。AC-DSGF 对同一预算日程仍敏感，同时保持低得多的 SCA。

90% 预算削减下约 11% 相对下降（平缓退化）。该平缓退化与**推论 1 的精神一致**：稀疏化引起的动作偏差仍保持有界，因为所学门控在激进预算削减下仍保留任务关键边。失败边界：

> **过度稀疏最终损害协同。** 当 \(\rho\to0\) 或通信惩罚过大时，剩余交互无法支撑协作；残差引导只能补偿有限的信息缺口，无法替代必要链路。

我们主张任务约束下的自适应稀疏——而非“通信越少越好”。

### 5.6 通信行为与拓扑一致性

SCA 与邻近风险正相关（约 +0.84）、与分散度负相关（约 −0.57）——风险条件化稀疏，而非固定速率通信。

在导航任务中，碰撞风险是协同的主导因素，故邻近相关较高。然而，AC-DSGF 的非对称性（有向门控；见表 2b）与任务阶段稳定性（Fig. 6）表明其学到的是**结构化、非互易**的稀疏模式，而非单纯距离衰减——例如信息性的领导—跟随链路可在相对距离变化时保持，而纯距离启发式更倾向于对称切换。

![Fig.6 行为](figures/Fig5_behavior.png)

**Fig. 6** 通信行为与任务情境相关（风险条件化稀疏）。

**拓扑一致性**（由门控稳定性升级）：冻结 rollout 显示 \(\mathrm{Var}(C_t)\approx3\times10^{-3}\)，且 \(>99\%\) 步满足 \(\Delta E_t=0\)。关键交互边在任务阶段内稳定复现，而非随机翻转——证据表明学习的是*与谁通信*，而非仅*少通信*。

### 5.7 丢包鲁棒性

| Method | 0% loss | 70% loss |
|--------|--------:|---------:|
| GAT | 21.2 | 15.8 |
| DSGF | 21.8 | 18.5 |
| AC-DSGF | 26.2 | 25.8 |

已稀疏的激活在随机边丢失下曲线更平坦。

### 5.8 规模与部署分析

与理论对齐的章节（非“附加实验”）。冻结 \(N{=}16\) 策略，零样本 \(N\in\{8,16,32\}\)。

#### 5.8.1 Fig. 7 — Communication Density Scaling

**Table 5** SCA \(C_s\)、归一化密度 \(\eta_N\) 与零样本 Success（%）。
*协议说明：* 64 episodes，seed 42（仅评估）。SCA 量级与表 1 因种子/rollout 随机性而有差异，但关键趋势 \(\eta_N=\mathcal{O}(1/N)\) 保持。\(N{=}32\) 时两方法 Success 仍非零——AC 以小幅 Success 代价换取数量级更低的 SCA。

| \(N\) | DSGF SCA | DSGF \(\eta_N\) | DSGF Succ. | AC SCA | AC \(\eta_N\) | AC Succ. |
|------|--------:|---------------:|-----------:|-------:|-------------:|---------:|
| 8 | 11.01 | 0.197 | 33.3 | 0.0020 | \(3.6\times10^{-5}\) | **40.6** |
| 16 | 22.62 | 0.094 | 19.8 | 0.0059 | \(2.5\times10^{-5}\) | **21.1** |
| 32 | 56.06 | 0.057 | 11.2 | 0.0188 | \(1.9\times10^{-5}\) | 10.3 |

![Fig.7 密度标度](../ac_dsgf/figures/Fig9_comm_density_scaling.png)

**Fig. 7** 通信密度随蜂群规模的变化。AC-DSGF 在 \(N\) 增大时保持近似恒定的稀疏激活密度，而 DSGF 交互密度显著更高——与 \(\eta_N=\mathcal{O}(K/N)\) 一致。主张：不同的通信增长规律，而非 \(N{=}32\) 时 Success 更高。

在 \(N=32\) 规模下，AC-DSGF 相较 DSGF 仅损失 **0.9 个百分点**的成功率，却将 SCA 降低了约 **2,982 倍**（56.06 → 0.0188）。这表明 \(\mathcal{O}(1/N)\) 规模律可直接转化为可部署的带宽节省，且不会造成任务灾难性失效。

#### 5.8.2 Fig. 8 — Learned Communication Selectivity

![Fig.8 选择性](../ac_dsgf/figures/Fig10_gate_distribution.png)

**Fig. 8** 所学通信选择性。（a）DSGF：半径支撑上 \(g\equiv1\)；（b）AC-DSGF：连续低强度激活，非全局静默。结构证据见 Top-\(K\) 排序（Fig. 9）。

#### 5.8.3 Fig. 9 — Top-\(K\) Deployment

**Table 6** Top-\(K\) 部署（\(N{=}16\)，eval-only）

| Mode | Success (%) |
|------|------------:|
| Soft gate | 25.8 |
| Top-1 | 24.5 |
| Top-2 | 28.1 |
| Top-3 | 29.4 |

![Fig.9 Top-K](../ac_dsgf/figures/Fig11_topk_deploy.png)

**Fig. 9** 软所学拓扑可离散化为实际通信链路，任务有效性相当。该经验观察由推论 2 提供理论支撑。去除弱软激活后的温和增益表明推理阶段存在不必要交互；我们**不**主张 Top-\(K\) 应取代软训练。


### 5.9 通信重要性分析

为表明门控**不是随机稀疏化**（与 Prop. 3 一致），我们冻结 AC-DSGF，按平均 \(g_{ij}\) 对等规模边队列（\(M{=}24\)）置零：top-\(g\)、bottom-\(g\) 与 random。

**Table 7** 队列删边（\(N{=}16\)）

| Cohort | Success | \(\Delta S\) (pp) |
|--------|--------:|------------------:|
| baseline | 27.3% | 0 |
| drop top-\(g\) | 24.0% | **−3.4**（伤害最大） |
| drop bottom-\(g\) | 27.6% | +0.3 |
| drop random | 31.0% | +3.7（无系统伤害） |

![Fig.10 边重要性](../ac_dsgf/figures/Fig12_edge_importance.png)

**Fig. 10** Edge Importance Consistency。删除高门控边损害任务完成；删除同等数量的低门控或随机边则否。*更高门控值与更大任务贡献相关联。*

### 5.10 通信—性能 Pareto

未重训多个 \(\lambda_c\)（v1 冻结），在同一 checkpoint 上扫描预算 / Top-\(K\) / 全开 / 随机，并以 DSGF/GAT 为锚点。

![Fig.11 Pareto](../ac_dsgf/figures/Fig13_comm_pareto.png)

**Fig. 11** 预算—性能工作曲线（冻结策略）：软 / 预算 / Top-\(K\) 干预下的 SCA vs Success。该图可视化 AC-DSGF 诱导的软预算权衡面，**不**声称多 \(\lambda_c\) 重训扫描。我们不主张 Success 领先；而是：

> AC-DSGF 在预算约束拓扑选择下提供 favorable operating region（可比 Success，SCA 远低于稠密基线）。

---

## 6 讨论

结果在仿真平台上获得，评估协议与硬件部署兼容。算法贡献与实现平台细节分离。局限：仿真环境；条件性规模/保持分析；过度稀疏下的失败边界。

---

## 7 结论

本文提出 **AC-DSGF**，面向多无人机蜂群在软预算下学习自适应通信稀疏模式的框架。在通信预算下，拓扑作为决策变量，通过 Candidate Edge Scoring、Budget-Constrained Edge Selection 与 Residual Recovery 构建。我们给出度约束下的通信规模界（\(\eta_N=\mathcal{O}(1/N)\)）、残差**动作稳定性**分析，并将所学门控解释为预算边选择的近似/诱导解（非显式组合求解器）。在相同通信预算下，AC-DSGF 以可扩展、鲁棒的拓扑适应取得具有竞争力的任务有效性。

---

## 参考文献

1. Yu C, et al. The surprising effectiveness of PPO in cooperative multi-agent games. NeurIPS, 2022.
2. Lowe R, et al. Multi-agent actor-critic for mixed cooperative-competitive environments. NeurIPS, 2017.
3. Rashid T, et al. QMIX: Monotonic value function factorisation. ICML, 2018.
4. Foerster J, et al. Learning to communicate with deep multi-agent RL. NeurIPS, 2016.
5. Foerster J, et al. Counterfactual multi-agent policy gradients. AAAI, 2018.
6. Sukhbaatar S, et al. Learning multiagent communication with backpropagation (CommNet). NeurIPS, 2016.
7. Das A, et al. TarMAC: Targeted multi-agent communication. ICML, 2019.
8. Jiang J, Lu Z. Learning attentional communication for multi-agent cooperation. NeurIPS, 2018.
9. Singh A, et al. Learning when to communicate at scale. ICLR, 2019.
10. Kim D, et al. Learning to schedule communication in MARL. ICLR, 2019.
11. Wang R, et al. Learning efficient multi-agent communication: An information bottleneck approach. ICML, 2020.
12. Veličković P, et al. Graph attention networks. ICLR, 2018.
13. Jiang J, et al. Graph convolutional reinforcement learning (DGN). ICLR, 2020.
14. Bettini M, et al. VMAS: A vectorized multi-agent simulator. arXiv:2207.03530, 2022.
15. Chung S-J, et al. A survey on aerial swarm robotics. IEEE T-RO, 2018.

---

## 附图清单

| Fig | 文件 | 内容 |
|-----|------|------|
| 0 | ../ac_dsgf/figures/Fig0_motivation_topology.png | 预设 → 任务驱动拓扑 |
| 1 | figures/Fig1_framework.png | 框架 |
| 2 | figures/Fig3_algorithm_flow.png | 流水线 |
| 3 | figures/Fig4_pareto.png | Success—SCA |
| 4 | figures/Fig7_comm_density.png | 预算饱和 |
| 5 | ../ac_dsgf/figures/Fig14_hard_topk.png | 固定硬 Top-\(K\) |
| 6 | figures/Fig5_behavior.png | 行为 |
| 7 | ../ac_dsgf/figures/Fig9_comm_density_scaling.png | 密度标度 |
| 8 | ../ac_dsgf/figures/Fig10_gate_distribution.png | 选择性 |
| 9 | ../ac_dsgf/figures/Fig11_topk_deploy.png | Top-\(K\) 部署 |
| 10 | ../ac_dsgf/figures/Fig12_edge_importance.png | 边重要性 |
| 11 | ../ac_dsgf/figures/Fig13_comm_pareto.png | 预算—性能工作曲线 |

文件名保留历史前缀；**显示图号按正文出现顺序 0–11**。
