# 将通信拓扑作为资源约束下多智能体协同的决策变量

**副标题：** 约束感知的动态图决策框架  
**方法实现复用：** AC-DSGF backbone（冻结，不新增模块）  
**轨道：** T-RO / TNNLS（B）· **v0.11** · *Submission Candidate · 科学内容冻结*  
**中心对象：** \(\boxed{G_t=\phi_\theta(s_t)}\)

> 本稿与 RA-L 终稿**独立**。未经在 DGDP 框架下重新推导，不得直接并入 RA-L 的 SCA 稀疏化叙事断言。

**全文符号冻结。**

| 符号 | 角色 | 禁止 |
|------|------|------|
| \(\phi_\theta\) | **拓扑策略** | 用 \(\pi_\theta\) 指代图学习器 |
| \(\pi_\psi\) | **物理动作策略** | 与拓扑打分混用 |
| \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\) | 联合协同策略 | 无参数的模糊 \(\Pi=(\pi,\phi)\) |
| \(\Pi_{B_t}\) | 预算**投影算子** | 称 \(\Pi_{B_t}(S_t)\) 为“最优拓扑” |
| \(G_t^\star\) | 投影前的全支撑**参考**拓扑 | “最优 / 最佳 / argmax 拓扑” |

---

## 摘要

多智能体协同策略常把通信图当作固定基础设施（半径邻域、\(k\)-NN 或全连接），只学习物理动作。在资源约束下，我们**将通信拓扑表述为可与物理策略联合优化的可学习协同决策变量**。我们形式化 *动态图决策过程*（DGDP），联合策略 \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\) 输出
\[
G_t=\phi_\theta(s_t),\qquad
a_t=\pi_\psi\bigl(s_t,\,M(\phi_\theta(s_t))\bigr),
\]
使图位于决策空间，而非仅位于特征流水线。可行拓扑由边效用估计与**预算投影算子** \(\Pi_{B_t}\) 得到。我们建立预算可行性与学习得分下的效用最大化投影（定理 1）——而非任务回报最优——以及共享状态下的信息 / 动作 / 回报差异界（Lemma 1–2、定理 2）与固定度数投影的复杂度命题。经验上，多 seed 孪生评估与 Lemma 1–2 一致；群体规模研究在固定 \(K\) 下呈现线性通信增长；信道压力测试与方法感知的 \((J,C)\) 比较完成实验程序。总体而言，该方法在**受限通信预算下实现通信高效协同**，算法实例复用冻结的 AC-DSGF 主干。

**关键词：** 动态图决策过程；可学习通信拓扑；预算投影；多智能体协同；资源约束

---

## 1 引言

### 1.1 拓扑作为基础设施 vs.\ 拓扑作为决策

大规模多智能体与多无人机系统中，通信受带宽、能量与干扰限制。常见做法把 \(G_t\) 当作**基础设施**：
\[
A_t = f_{\mathrm{geo}}(x_t),
\]
再只学习 \(a_t=\pi(s_t)\)。

我们改为把图放入**动作 / 决策空间**：
\[
\boxed{G_t = \phi_\theta(s_t)},
\qquad
a_t = \pi_\psi\bigl(s_t,\,M(G_t)\bigr).
\]
联合对象为 \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\)。

**主 claim（不可弱化或过强）。**  
将通信拓扑表述为可与物理策略联合优化的可学习协同决策变量。  
**不**声称在 \(\max_G J(\pi,G)\) 意义上“优化通信拓扑”；学习得分进入在 \(\mathcal{G}_{B_t}\) 上最大化 \(\sum g_{ij}\) 的投影。

### 1.2 与通信 / 图 MARL 的差距

既有通信 MARL 主要优化*消息内容*、*门控*或*注意力聚合*（如 CommNet、TarMAC、ATOC、IC3Net），支撑图多为固定或启发式限制。图方法（如 GAT、DGN）提供局部 / 关系型协同表征，但通常把 \(G\) 当作给定基础设施。另一方面，带宽、能量与干扰使通信成为**显式资源约束**——更接近约束决策，而非无约束消息学习。

| 路线 | 典型对象 | 拓扑角色 |
|------|----------|----------|
| 消息学习（CommNet、TarMAC 等） | 消息内容 / 注意力 | 支撑图大体固定 |
| 图 MARL（GAT、DGN 等） | 在 \(G\) 上聚合 | \(G\) 默认半径/KNN |
| 事后稀疏化 | 剪枝固定 \(G\) | 与任务耦合弱 |
| **本文** | \(G_t=\phi_\theta(s_t)\) 作为决策 | 预算投影 + 联合 \(\Pi_{\theta,\psi}\) |

### 1.3 贡献

1. **拓扑作为决策。** 将通信拓扑表述为在显式资源约束下与物理策略联合优化的可学习协同变量（DGDP + \(\Pi_{\theta,\psi}\)）。  
2. **理论。** 建立预算约束拓扑投影的可行性（定理 1），以及连接拓扑近似、信息差异与动作偏差的共享状态稳定性界（Lemma 1–2、定理 2）。  
3. **通信复杂度。** 刻画固定度数拓扑投影下的通信复杂度：当 \(K=O(1)\) 时 \(C=O(N)\)（复杂度命题）——**不是**图规模性能定理。  
4. **实验。** 报告约束与不完美通信下的通信–性能权衡：预算可行性、孪生理论证据、经验线性通信增长、信道压力测试与方法感知的 \((J,C)\) 基线。

**主一句 claim。** 带预算约束的拓扑学习，在固定度数投影下具有**线性通信增长**。  
*（禁止贡献措辞：可扩展通用智能；最优协同；普适拓扑学习。）*

---

## 2 相关工作

**通信学习 MARL。** CommNet、DIAL、TarMAC、ATOC、IC3Net 主要学习 *何时/发什么* 或 *向谁注意*，支撑图多为固定或启发式限制；硬可行集 \(\mathcal{G}_{B_t}\) 通常不是一等公民（[`references/communication_marl.md`](references/communication_marl.md)）。

**图 MARL。** GAT/DGN 等在图上学习聚合器，提供局部关系归纳偏置；图本身很少作为与控制联合优化的约束决策变量（[`references/graph_marl.md`](references/graph_marl.md)）。

**资源约束 / 通信高效 RL。** CMDP 与约束策略优化强调硬资源限制；事件触发与事后剪枝降低*给定图*上的代价，但不定义到 \(\mathcal{G}_{B_t}\) 的投影（[`references/constrained_rl.md`](references/constrained_rl.md)）。

**协同设定。** 合作 MARL 与多机器人仿真（MAPPO、VMAS 等）在假定交互结构下学习物理动作（[`references/multi_agent_coordination.md`](references/multi_agent_coordination.md)）。相邻结构学习 / 稀疏化不采用本稿 DGDP + 硬 \(\Pi_{B_t}\) 表述（[`references/related_topology_learning.md`](references/related_topology_learning.md)）。

**区分。** 我们将 \(G_t=\phi_\theta(s_t)\) 提升为在 \(C(G_t)\le B_t\) 下的**可学习协同决策**，经 \(\Pi_{B_t}\) 保证可行性，并分析拓扑诱导的信息 / 动作差异——**不**声称投影最大化任务回报。实验比较的是不同**通信代价体制**的代表范式，而非注意力机制大比武。

---

## 3 动态图决策过程

### 3.1 定义 1（DGDP）

\[
\mathcal{M}_G
=
\bigl(\mathcal{S},\mathcal{A},\mathcal{G},P,R,C,\gamma\bigr).
\]

**决策结构：**
\[
G_t=\phi_\theta(s_t),\qquad
m_t=M\bigl(\phi_\theta(s_t)\bigr),\qquad
a_t=\pi_\psi(s_t,m_t).
\]

**拓扑–动作耦合（紧凑形式）：**
\[
a_t
=
\pi_\psi\bigl(s_t,\,\phi_\theta(s_t)\bigr)
\quad\text{（经消息映射 \(M\)）},
\qquad
\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta).
\]

故 \(G_t\) 是**决策的一部分**，而非仅 \(\pi_\psi\) 的输入特征。

### 3.2 预算：静态与动态

\[
C(G_t)\le B_t.
\]
- **动态预算** \(B_t\)：时变容量。  
- **静态预算** \(B\)：特例 \(B_t\equiv B\)。

\[
\mathcal{G}_{B_t}=\{G\in\mathcal{G}:C(G)\le B_t\}.
\]

### 3.3 联合拓扑–策略优化

\[
\boxed{
\max_{\theta,\psi}
J(\Pi_{\theta,\psi})
\quad
\mathrm{s.t.}
\quad
G_t=\phi_\theta(s_t),\;
C(G_t)\le B_t.
}
\]

软训练的 \(\lambda\) 惩罚为拉格朗日松弛；硬可行性由 \(\Pi_{B_t}\) 保证。

### 3.4 实例化（主干冻结）

v0.2 不引入新模块；贡献为 DGDP、符号冻结与投影定理。

---

## 4 约束感知拓扑投影

### 4.1 边效用学习器

\[
S_t=\phi_\theta(s_t),\qquad g_{ij}=\sigma(s_{ij})\cdot A_{t,ij}.
\]

### 4.2 投影算子 \(\Pi_{B_t}\)

\[
G_t=\Pi_{B_t}(S_t)=\Pi_{B_t}\bigl(\phi_\theta(s_t)\bigr).
\]

**措辞锁定。** 称 \(G_t\) 为学习边效用下的**预算可行效用最大化投影**。禁止称“最优拓扑”或“任务最优通信图”。保持算子抽象，主文不展开 sorting / knapsack。

### 4.3 定理 1 — 预算约束动态拓扑投影的可行性

**定理 1（预算约束动态拓扑投影的可行性）。**  
对任意 \(S_t\)，\(G_t=\Pi_{B_t}(S_t)\) 满足：

**(1) 可行性：** \(C(G_t)\le B_t\)。  

**(2) 效用最大化投影（非任务最优）：** 对任意 \(G'\in\mathcal{G}_{B_t}\)，
\[
\sum g_{ij}x^\star_{ij}\ge\sum g_{ij}x'_{ij},
\]
且**不**断言 \(G_t\in\arg\max_{G} J(\pi_\psi,G)\)。

详见 [`theory/theorem_budget.md`](theory/theorem_budget.md)。

---

## Algorithm 1 — 约束感知动态图决策

```
Require:
  state s_t, budget B_t, candidate support A_t
  topology policy φ_θ, physical policy π_ψ, message map M

1  S_t ← φ_θ(s_t)
2  G_t ← Π_{B_t}(S_t)          # 约束投影算子（不展开 solver）
3  m_t ← M(s_t, G_t)
4  a_t ← π_ψ(s_t, m_t)
5  Execute
```

---

## 5 理论分析

建立从预算拓扑决策到共享状态回报控制的三环节链：
\[
\boxed{
\text{Budget}
\;\rightarrow\;
\text{Topology distance}
\;\rightarrow\;
\text{Information loss}
\;\rightarrow\;
\text{Return bound}
}
\]
即 \(\textbf{约束可行性}+\textbf{拓扑诱导的性能稳定性}\)。  
完整证明：[`theorem_budget.md`](theory/theorem_budget.md)、[`lemma_information_discrepancy.md`](theory/lemma_information_discrepancy.md)、[`theorem_return_bound.md`](theory/theorem_return_bound.md)。

### 5.1 动态拓扑投影可行性

**定义（可行图集）。** \(\mathcal{G}_{B_t}=\{G:C(G)\le B_t\}\)；静态预算为特例 \(B_t\equiv B\)。

**定理 1（预算约束动态拓扑投影的可行性）。**  
\(S_t=\phi_\theta(s_t)\)，\(G_t=\Pi_{B_t}(S_t)\) 满足：(1) \(C(G_t)\le B_t\)；(2) 在 \(\mathcal{G}_{B_t}\) 上最大化 \(\sum g_{ij}\)—**不是** \(\arg\max_G J\)。

### 5.2 拓扑诱导信息差异

**全信息通信参考**（非最优）：\(G_t^\star=G_{\mathrm{full}}\)。  
图距离：\(d_G=\|A_t-A_t^\star\|_F\)。

**假设 M。** \(\|M(G_1)-M(G_2)\|\le L_M\,d_G(G_1,G_2)\)。

**Lemma 2（拓扑稀疏化诱导有界信息差异）。**  
对定理 1 的 \(G_t=\Pi_{B_t}(S_t)\)，
\[
\boxed{
\varepsilon_G(t)=\|M(G_t^\star)-M(G_t)\|
\le L_M\|A_t-A_t^\star\|_F.
}
\]
有限 \(N\) \(\Rightarrow\) 有限 \(\varepsilon_G\)。Lemma 2 **桥接** Thm.~1 与 Thm.~2；不声称 \(\varepsilon_G\) 任务最优或趋于零。

### 5.3 共享状态性能稳定性

**共享状态（孪生）评估。** 固定 \(\{s_t\}\)，定义 \(J_T^\star,J_T\)（固定轨迹耦合下的折扣累积奖励；不声称 \(\rho^\star\approx\rho\)）。

**Lemma 1.** \(\|a_t^\star-a_t\|\le L_\pi\varepsilon_G\)。

**定理 2（共享状态拓扑近似下的折扣回报界）。**  
\[
\boxed{
|J_T^\star-J_T|
\le
\frac{L_R L_\pi\varepsilon_G}{1-\gamma}
\le
\frac{L_R L_\pi L_M}{1-\gamma}\sup_t d_G(G_t,G_t^\star).
}
\]
该界刻画共享状态耦合下的拓扑诱导退化；闭环保证需要额外转移正则性假设——本稿**不**引入。

**推论 1（残差恢复）。**  
\[
\|a^\star-a\|\le L_\pi\varepsilon_G+\beta\varepsilon_\Delta,
\qquad
|J^\star-J|
\le
\frac{L_R(L_\pi\varepsilon_G+\beta\varepsilon_\Delta)}{1-\gamma}.
\]

### 5.4 命题 — 固定度数投影下的通信复杂度

**命题（固定度数投影下的通信复杂度）。**  
若 \(G=\Pi_B(S)\) 满足 \(\max_i d_i(G)\le K\)，则 \(\lvert E(G)\rvert\le NK\)，从而 \(C(G)=O(NK)\)。当 \(K=O(1)\) 时，
\[
C(G)=O(N).
\]
**证明。** 对各智能体度数上界求和。□  

此为定理 1 度数预算特例的直接推论；**不**断言跨规模性能保持或泛化。  
详写：[`theory/proposition_complexity.md`](theory/proposition_complexity.md)。  
*（强形式定理 3 已取消 — 见 [`theory/theorem_scalability.md`](theory/theorem_scalability.md)。）*

### 5.5 理论状态

| 编号 | 角色 | 状态 |
|------|------|------|
| 定理 1 | 预算可行性 + 效用最大化投影 | **冻结** |
| Lemma 2 | 投影 \(\rightarrow\) 有界 \(\varepsilon_G\) | **已并入** |
| Lemma 1 / 定理 2 / 推论 1 | 共享状态回报稳定性 | **冻结** |
| 命题 | 固定度数投影下的通信复杂度 | **冻结** |
| 定理 3 | 图规模性能 / 泛化 | **已取消** |

## 6 实验

**状态：** 设计已冻结 — 见 [`experiments/tro_experimental_design.md`](experiments/tro_experimental_design.md)。  
**实现协议：** [`tro_logging_protocol.md`](experiments/tro_logging_protocol.md) 等。  
**§6.3：已冻结**（[`evidence_6_3_formal/`](experiments/evidence_6_3_formal/)）。实验报告与理论*一致*的趋势，不证明引理/定理。

组织原则：理论**对齐**的经验考察
\[
\phi_\theta(s)\;\rightarrow\;G_t=\Pi_{B_t}\;\rightarrow\;C\le B
\;\rightarrow\;D_G\;\rightarrow\;\varepsilon_G\;\rightarrow\;\Delta A_\gamma.
\]

### 6.1 实验设置

任务 T1（§6.3 使用）/ T2 / T3；冻结 AC-DSGF+MAPPO；基线 A/B/C（§6.6）；指标含 \(V_B,D_G,\varepsilon_G,\Delta A_\gamma,J,C\)。

### 6.2 预算约束拓扑投影分析
**状态：已冻结（正式表）。**

*支撑定理 1（可行性——非任务性能）。*  
协议：[`experiments/tro_6_2_formal_protocol.md`](experiments/tro_6_2_formal_protocol.md)。  
考察硬投影 \(G_t=\Pi_{B_t}(S_t)\) 是否在不同规模与预算参数化下始终满足 \(C(G_t)\le B_t\)。本节**不是**性能对比（见 §6.3 / §6.6）。

**因子。** \(N\in\{8,16,32,64\}\)；固定度数 \(K\in\{1,2,4,6,8\}\)；比例预算 \(\rho_B\in\{0.05,0.1,0.2,0.4\}\)；五 seed 冻结 `uav16`；不训练。\(N\le 16\) 用闭环 VMAS rollout；\(N\in\{32,64\}\) 对同一冻结打分器 + 硬 \(\Pi_{B_t}\) 做几何位置采样（投影算子不变）。

**指标。** \(VR\)、\(V_{\max}\)、\(\bar d\)、\(\rho_t\)。固定-\(K\) 下可行上界为 \(k_i=\min(K,\lvert\mathcal{N}_i\rvert)\)；当半径邻域小于 \(K\) 时，\(\bar d\) 跟随 \(\overline{k_{\mathrm{cap}}}\) 而非无约束 \(K\)。

**结果。** 全部 \((N,K)\)、\((N,\rho_B)\) 与五 seed 上 \(V_{\max}=0\)、\(VR=0\)。完整网格：[`evidence_6_2_budget/`](experiments/evidence_6_2_budget/)。Fig.~4 展示固定 \(K\) 下 \(\rho\) 随 \(N\) 下降（不作可扩展性断言）。

**表 I。** 不同规模与约束下的预算可行性（seed 聚合；代表性 fixed-\(K\) 切片）。所有 ratio 单元同样 \(V_{\max}=VR=0\)。

| \(N\) | \(K\) | 预算类型 | 平均度数 | 最大违约 | 违约率 |
|------:|------:|----------|---------:|---------:|-------:|
| 8 | 2 | fixed-\(K\) | 1.023 | 0 | 0 |
| 16 | 4 | fixed-\(K\) | 1.209 | 0 | 0 |
| 32 | 6 | fixed-\(K\) | 1.352 | 0 | 0 |
| 64 | 8 | fixed-\(K\) | 1.285 | 0 | 0 |

> 投影层在不同群体规模与约束设定下始终满足预定通信预算，表明所提约束拓扑决策机制在实现层面具有可行性。

### 6.3 拓扑诱导信息与性能分析
**状态：已冻结。**

我们从经验上考察学习拓扑是否呈现理论所刻画的关系，即拓扑偏离、信息差异与动作差异之间的联系。全文中 \(G_t^\star\) 表示**预算投影前的全支撑参考拓扑**（几何支撑上的投影前分数），\(G_t=\Pi_{B_t}(S_t)\) 为投影后的稀疏拓扑。评估使用五个冻结 `uav16` checkpoint（seeds \(1234,2026,3407,42,8888\)），\(K\in\{1,2,4,6\}\)（避免半径邻域饱和），每 seed \(32\) episode。图件：[`figures/Fig1_topology_information.png`](figures/Fig1_topology_information.png)、[`Fig2_information_action.png`](figures/Fig2_information_action.png)、[`Fig3_budget_performance.png`](figures/Fig3_budget_performance.png)。

**拓扑–信息关系（Fig.~1）。** 更强稀疏化（更小 \(K\)）增大拓扑偏离 \(D_G=\|A_t-A_t^\star\|_F\)。消息差异 \(\varepsilon_G=\|M(G_t^\star)-M(G_t)\|\) 近似线性跟随 \(D_G\)：\(\varepsilon_G\approx\alpha D_G+\beta\)（\(K\)-均值上 \(\alpha\approx0.72\)，\(R^2\approx0.94\)）。该经验关系与 Lemma~2 所刻画的拓扑诱导信息差异 **一致**；我们不声称对 \(L_M\) 的统计识别。

**信息–动作关系（Fig.~2）。** 在共享状态孪生评估（E1）下，我们度量瞬时动作差 \(\Delta a=\|a^\star-a\|\) 与折扣动作差异
\[
\Delta A_\gamma=\sum_{t}\gamma^t\|a_t^\star-a_t\|.
\]
二者均随 \(K\) 增大、\(\varepsilon_G\) 缩小而单调下降，与 Lemma~1 **一致**。  
由于孪生评估在相同状态下隔离拓扑诱导的信息变化，我们以折扣动作差异作为主指标，而非额外依赖闭环状态分布漂移的任务回报差。该选择使 Fig.~2 对齐定理 2 的证明中间量，同时不过度声称闭环回报保持。

**预算–性能权衡（Fig.~3）。** 闭环（E0）下 reward、success 与通信代价相对预算水平（固定-\(K\) 与比例调度；五 seed mean±std）呈现自适应通信–性能 Pareto：在当前导航设定下，激进稀疏化显著降低 \(C\)，而任务回报仍与全支撑参考处于可比范围。Fig.~3 服务于理论叙事中的权衡说明；统计方法比较留待 §6.6。

### 6.4 群体规模增大下的可扩展性分析
**状态：已冻结（经验分析；无定理 3）。**

协议：[`experiments/tro_6_4_formal_protocol.md`](experiments/tro_6_4_formal_protocol.md)。  
证据：[`experiments/evidence_6_4_scalability/`](experiments/evidence_6_4_scalability/)。  
在**固定**度数预算 \(K=4\) 下考察通信代价与效率随群体规模的变化（不重新训练；不随 \(N\) 重调 \(K\)）。本节为**经验可扩展性分析**：支撑 §5.4 复杂度命题，**不**建立性能泛化定理。

**因子。** \(N\in\{16,32,64,128\}\)；方法 AC-DSGF（\(K=4\)）、DSGF、Full Attention；五冻结 seed；每格 16 episode。生成几何随 \(N\) 放大；动力学 / 奖励 / \(K\) 固定。

**表述纪律。** 回报 \(J_N\) 随 \(N\) 上升主要因任务总量尺度变化；成功率随 \(N\) 下降反映难度变化。因此**不**声称“性能随规模扩展”或“拓扑学习可从 \(N=16\) 泛化到 \(128\)”。主结论面向通信增长与效率。

**观察 1 — 近似线性通信增长。**  
AC-DSGF 的 \(C_N/N\) 在 \(N:16\to128\) 上保持窄带 \(\approx 1.54\)–\(1.93\)，与固定 \(K\) 下 \(C_N=O(N)\)（及复杂度命题）一致。DSGF 类似（\(\approx 1.47\)–\(2.10\)）。

**观察 2 — 稀疏通信优势。**  
Full Attention 代价为 \(C=N(N-1)\)（\(C/N:15\to127\)），近二次增长。\(N=128\) 时 AC-DSGF 均值代价约 \(247\)，Full Attention 为 \(16256\)（Fig.~7）。

**观察 3 — 通信效率。**  
\(\eta_N=J_N/C_N\) 上稀疏方法显著占优：AC-DSGF 由 \(\approx 0.65\) 升至 \(\approx 1.16\)，Full Attention 由 \(\approx 0.07\) 降至 \(\approx 0.016\)（Fig.~8）。此为通信效率证据，而非任务回报可扩展性定理。

**表 III。** seed 聚合可扩展性统计（5 seeds 均值；AC-DSGF 固定 \(K=4\)）。

| 方法 | \(N\) | \(J\) | Success | \(C\) | \(C/N\) | \(\eta=J/C\) |
|------|------:|------:|--------:|------:|--------:|-------------:|
| AC-DSGF (\(K=4\)) | 16 | \(15.90\pm0.97\) | 0.212 | \(24.6\pm1.9\) | 1.54 | 0.651 |
| AC-DSGF (\(K=4\)) | 32 | \(41.08\pm3.93\) | 0.105 | \(60.4\pm6.8\) | 1.89 | 0.685 |
| AC-DSGF (\(K=4\)) | 64 | \(117.07\pm9.30\) | 0.052 | \(111.5\pm5.7\) | 1.74 | 1.052 |
| AC-DSGF (\(K=4\)) | 128 | \(287.00\pm25.57\) | 0.026 | \(247.2\pm14.6\) | 1.93 | 1.163 |
| DSGF | 16 | \(16.16\pm1.01\) | 0.202 | \(23.6\pm2.4\) | 1.47 | 0.690 |
| DSGF | 32 | \(41.09\pm2.96\) | 0.108 | \(60.2\pm7.4\) | 1.88 | 0.689 |
| DSGF | 64 | \(118.30\pm5.23\) | 0.049 | \(118.4\pm15.4\) | 1.85 | 1.012 |
| DSGF | 128 | \(289.08\pm17.40\) | 0.025 | \(269.3\pm32.5\) | 2.10 | 1.085 |
| Full Attention | 16 | \(16.82\pm1.86\) | 0.243 | \(240\) | 15.0 | 0.070 |
| Full Attention | 32 | \(41.45\pm6.21\) | 0.122 | \(992\) | 31.0 | 0.042 |
| Full Attention | 64 | \(115.47\pm15.79\) | 0.045 | \(4032\) | 63.0 | 0.029 |
| Full Attention | 128 | \(257.92\pm35.60\) | 0.016 | \(16256\) | 127.0 | 0.016 |

图件：[`figures/Fig7_communication_scaling.png`](figures/Fig7_communication_scaling.png)（\(C_N\) vs \(N\)）；[`figures/Fig8_efficiency_scaling.png`](figures/Fig8_efficiency_scaling.png)（\(\eta_N\) vs \(N\)）。

> 在固定度数预算下，学习的稀疏拓扑呈现近似线性的通信增长，且通信效率显著高于稠密注意力（经验观察）——但不声称跨群体规模的闭环性能泛化。

### 6.5 不完美通信信道下的性能表现
**状态：已冻结。**

协议：[`experiments/tro_6_5_formal_protocol.md`](experiments/tro_6_5_formal_protocol.md)。  
冻结策略、不重新训练（5 seeds × 16 episode）。Fig.~6：[`figures/Fig6_channel_robustness.png`](figures/Fig6_channel_robustness.png)（文件名沿用；主张为性能稳定性，非鲁棒性定理）。

**方法。** AC-DSGF；DSGF；Full Attention。

**丢包。** \(p_\ell\in\{0,0.1,0.3,0.5\}\) 下，AC-DSGF 回报保持在窄带（\(J\approx 8.9\)–\(9.3\)），交付代价随丢包下降（\(C:43\rightarrow22\)），success 约 \(0.23\)。DSGF 类似；Full Attention 绝对回报更高但 \(C\in[120,240]\)。

**时延。** \(\tau\in\{0,20,50,100\}\) ms 下三者 \(J\) 仅轻微波动，未见急剧坍塌。

**带宽。** \(B/B_{\mathrm{full}}\in\{0.1,0.2,0.4,0.8\}\) 硬投影下，AC-DSGF 回报约 \(J\approx9.1\)，而 \(C\) 随预算上升（\(15\rightarrow42\)），与定理 1 的可行集一致。

> AC-DSGF 在通信条件退化时仍能保持相对更稳定的协同性能。

输出：[`evidence_6_5_channel/`](experiments/evidence_6_5_channel/)。

### 6.6 与通信学习基线对比
**状态：** Phase A **已冻结**（\(C\) 定义已校正）。

协议：[`experiments/tro_6_6_formal_protocol.md`](experiments/tro_6_6_formal_protocol.md)。  
在匹配 \(N=16\)、相同 5 seeds×32 episode 下，比较对应不同**通信代价体制**的代表性范式，于联合 \((J,C)\) 平面评估。Fig.~5：[`figures/Fig5_baseline_pareto.png`](figures/Fig5_baseline_pareto.png)。

**Phase A（代价体制）。**  
(A) MAPPO — 无学习通信（\(C=0\)）；  
(B) GAT-MAPPO、DSGF — 半径支撑聚合，无预算投影决策；  
(G2) Full Attention — 稠密上界（\(C=N(N-1)=240\)）；  
(Ours) AC-DSGF — 硬 \(\Pi_{B_t}\)，含固定 \(K\in\{2,4,6\}\)。

**表 II.** 五 seed mean±std（统一 reeval）。

| 类 | 方法 | \(J\) | Success | \(C\) |
|----|------|------:|--------:|------:|
| A | MAPPO | \(9.62\pm0.64\) | \(0.272\) | \(0\) |
| B | GAT-MAPPO | \(8.78\pm0.52\) | \(0.221\) | \(44.4\pm2.4\) |
| B | DSGF | \(9.01\pm0.42\) | \(0.232\) | \(43.7\pm3.6\) |
| G2 | Full Attention | \(9.60\pm0.90\) | \(0.288\) | \(240\) |
| Ours | AC-DSGF（默认） | \(8.96\pm0.66\) | \(0.227\) | \(43.5\pm1.3\) |
| Ours | AC-DSGF (\(K=2\)) | \(9.24\pm0.68\) | \(0.239\) | \(27.5\pm0.5\) |
| Ours | AC-DSGF (\(K=4\)) | \(9.14\pm0.73\) | \(0.254\) | \(40.9\pm1.1\) |
| Ours | AC-DSGF (\(K=6\)) | \(9.26\pm0.58\) | \(0.234\) | \(43.6\pm2.0\) |

**解读。** AC-DSGF 在 \(K=2\) 时回报与 Class B 相当，而通信低于半径支撑基线；Full Attention 代价高一个数量级，相对 MAPPO 回报增益有限。比较轴是**预算约束下的拓扑决策**，而非注意力机制竞赛；不声称优于所有消息学习方法。

## 7 讨论（预览）

将拓扑提升为决策变量，形成约束可行性 + 拓扑诱导性能稳定性。中心公式仍为
\[
\boxed{G_t=\phi_\theta(s_t)},
\]
经效用估计与硬投影 \(\Pi_{B_t}\) **在线**得到——**不是**对固定图的训练后 / 事后剪枝。固定度数投影下的近似线性通信增长（§6.4）与复杂度命题一致——**不**声称跨规模性能泛化；强形式定理 3 已取消。

---

## 参考文献

分类种子列表见英文稿 §References；文献笔记：[`references/`](references/)。