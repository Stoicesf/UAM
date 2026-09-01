# 多无人机协同搜索中的自适应通信拓扑优化方法研究

## AC-DSGF：将通信拓扑选择建模为可学习决策

**汇报性质：** 科研进展 
**核心一句话：**

> **AC-DSGF 将通信拓扑选择建模为可学习决策，在有限通信预算下实现多无人机协同任务性能保持与通信成本降低。**

**建议顺序：** 问题 → 不足 → 方法 → 主实验 → Pareto/Budget/Packet Loss → Demo 

材料速查：`paper/ac_dsgf/AC_DSGF_v1.pdf` · `paper/ac_dsgf/figures/` · `demo/videos/ac_dsgf_dynamic_comm.mp4`

---

# 1. 研究背景与问题定义

## 1.1 研究背景

多无人机系统在搜索、巡检、灾害响应等任务中，需要通过通信完成协同决策。

随规模增大：

- 链路数可至 \(O(N^2)\)
- 带宽 / 能耗 / 延迟受限
- **固定拓扑**难以适应动态任务需求

| 方法 | 主要问题 |
|------|----------|
| 全连接通信 | 通信成本高 |
| 固定邻居 / 半径图 | 缺乏任务适应性 |
| 随机丢弃消息 | 非任务相关 |
| GAT 类图通信 | 学习关系，但通常不考虑通信预算 |

**缺口：** 优化“如何决策”多，优化“如何通信”少。

**问题：** 能否把通信拓扑选择本身变成可学习决策变量？

---

### 【图1】研究问题示意

**【建议画面】**

![fig_pptx_problem_scene](F:\UAM\paper\figures\fig_pptx_problem_scene.png)

![fig_pptx_method_compare](F:\UAM\paper\figures\fig_pptx_method_compare.png)

![graph_attention_gat](F:\UAM\demo\figures\graph_attention_gat.png)

<video src="F:\UAM\demo\videos\ac_dsgf_dynamic_comm.mp4"></video>





| 左侧 Full Communication | 右侧 Adaptive Communication |
|-------------------------|-----------------------------|
| 全连接密边 | 仅关键边：U1—U2，U4—U2 |

**口头：** 目标不是“少发消息”，而是“**需要时才通信**”。

（可用 Demo 视频截帧对照；正式示意图可用 `framework.png` 前半对比页。）

---

# 2. 前期发现：为什么需要 AC-DSGF

## 2.1 长时域引导退化（机制动机）

已有图通信可改善短期协调，但若 **引导过度支配策略**，会出现长时域任务能力崩溃。

**已核实消融（4 UAV，勿与 16 UAV 主表混比）：**

| 变体 | Success |
|------|---------|
| Full DSGF（残差） | **9.27%** |
| **w/o Residual** | **0.22%**（约 42× 下降） |

说明：不是“再堆通信网络”，而是必须 **残差解耦 + 通信可学可控**。

> Reward 优化 ≠ 任务泛化；通信与引导机制需要重新设计。

---

### 【图2】动机图

| 文件 | `paper/ac_dsgf/figures/fig_motivation_degradation.png` |
| ---- | ------------------------------------------------------ |
|      |                                                        |

![fig_motivation_degradation](F:\UAM\paper\figures\fig_motivation_degradation.png)| 含义 | 去掉残差 → 性能崩溃；AC 在保留残差的同时学习通信拓扑 |
| 展示 | 标红对比柱；说明“为何改通信机制，而非调几个 λ” |

> **注意：** 汇报请使用上述已核实数据。勿使用未在本仓正式固化的 “GAT 20k→102k：4.74%→0.22%” 说法（0.22% 对应的是残差消融，不是 GAT 长训曲线）。

---

# 3. AC-DSGF 方法设计

## 3.1 核心思想

将通信拓扑：

- 从：**环境固定变量**
- 变为：**策略可学习决策** \(G_t=f(o_t)\)

智能体自主学习：**是否通信 / 与谁通信 / 通信强度**。

---

## 3.2 整体框架

### 【图3】框架图（汇报核心页）

| 文件 | `paper/ac_dsgf/figures/framework.png` |
| ---- | ------------------------------------- |
|      |                                       |

![framework](F:\UAM\paper\ac_dsgf\figures\framework.png)

```text
Observation
    → Dynamic / Quality Graph
    → Communication Gate  g_ij
    → Budget Layer (Top-K)
    → Residual DSGF Policy
    → Action
```

### Module 1：Adaptive Communication Gate

\[
g_{ij}=\sigma\!\big(W[h_i,h_j,d_{ij},r_{ij}]\big)\in[0,1]
\]

软概率强度（Notation 中已强调：不是默认二值边）。

### Module 2：Communication Budget

\[
A^{\mathrm{AC}}=A\odot E^{K}
\]

保证通信成本可控。

### Module 3：Residual DSGF Policy

\[
a=\pi(o)+\beta\Phi
\]

通信**辅助**策略，不替代策略。

---

# 4. 实验环境

| 参数 | 设置 |
|------|------|
| 环境 | VMAS Multi-Agent Navigation |
| 规模 | 主表 **16 UAV**；机制消融 **4 UAV**（分表） |
| 算法 | MAPPO / GAT / Transformer / DSGF / AC-DSGF |
| 训练 | 102k frames |
| Seeds | **5** |
| 评价 | 主表 **200 episodes** |
| 指标 | Success / Comm / CEI（+ Budget / Packet Loss 等） |

---

# 5. 实验结果 1：总体性能（主证据）

## 【表1】16 UAV 主实验（5 seeds）

| Method | Success (%) | Comm |
|--------|-------------|------|
| MAPPO | 2.40±0.90 | 0 |
| GAT | 2.04±0.90 | ~39 |
| DSGF | **4.01±1.68** | 39.27 |
| **AC-DSGF** | **3.95±0.83** | **0.43** |

**标记：**

- Success：**接近 DSGF**（可比，非宣称第一）
- Comm：**39.27 → 0.43（约 91×）**

数据：`paper/tables/table1_final.csv`

---

### 【图4】Success–Communication Pareto

| 文件 | `paper/ac_dsgf/figures/fig_comm_pareto.png` |![fig_comm_pareto](F:\UAM\paper\figures\fig_comm_pareto.png)
| 期望印象 | AC-DSGF 在**低通信、可比 Success** 区域 |

---

# 6. 实验结果 2：通信预算

**问题：** 限制预算会不会性能塌方？

## 【表2】Budget Sweep（16 UAV，诊断协议）

| Budget | GAT | DSGF | AC-DSGF |
|--------|-----|------|---------|
| 100% | 21.3 | 21.7 | **27.1** |
| 50% | 18.2 | 21.6 | **25.2** |
| 10% | 16.9 | 21.7 | **24.2** |

**结论：** 预算大幅下降，AC 任务表现保持（*graceful degradation*）。  
口头注明：诊断 ep 数与主表不同，看**趋势**。

### 【图5】`fig_comm_budget16.png`![fig_comm_budget16](F:\UAM\paper\ac_dsgf\figures\fig_comm_budget16.png)

---

# 7. 实验结果 3：Silence Collapse

**问题：** 是不是“学废了直接关通信”？

| Variant | Success | Comm |
|---------|---------|------|
| AC-full | 24.6 | **0.008** |
| No Budget（\(g=A\)） | 28.9 | **40.6** |
| Random | 28.5 | 20.3 |

打开预算后门控，通信暴涨、Success 仅小幅变化 → **不是 silence collapse**。

### 【图6】`fig_comm_ablation.png`![fig_comm_ablation](F:\UAM\results\ac_dsgf\comm_ablation\fig_comm_ablation.png)

---

# 8. 实验结果 4：通信行为分析

| 因素 | corr(Comm, ·) |
|------|----------------|
| 邻近风险 | **+0.84** |
| 空间分散 | −0.57 |
| 目标距离 | ≈0 |

+ Gate stability：\(>99\%\) 步 \(\Delta E=0\) → **任务相关，非随机开关**。

### 【图7】`fig_comm_trigger.png`（辅：`fig_gate_stability.png`）

![fig_comm_trigger](F:\UAM\paper\ac_dsgf\figures\fig_comm_trigger.png)

![fig_gate_stability](F:\UAM\results\ac_dsgf\gate_stability\fig_gate_stability.png)

---

# 9. 实验结果 5：鲁棒性（Packet Loss）

| Method | 0% | 70% |
|--------|----|-----|
| GAT | 21.2 | 15.8 |
| **AC-DSGF** | **26.2** | **25.8** |

GAT 明显下滑；AC **几乎持平** → *graceful degradation under unreliable communication*。

### 【图8】`fig_packet_loss.png`

![fig_packet_loss](F:\UAM\results\ac_dsgf\packet_loss\fig_packet_loss.png)

---

# 10. 工程 Demo

**目的：** 展示“按任务状态调通信”，**不是**刷最高 Success。

| 视频 | `demo/videos/ac_dsgf_dynamic_comm.mp4` |
| ---- | -------------------------------------- |
|      |                                        |

<video src="F:\UAM\demo\videos\ac_dsgf_dynamic_comm.mp4"></video>

| 曲线 | `demo/figures/communication_evolution.png` |
| ---- | ------------------------------------------ |
|      |                                            |

![communication_evolution](F:\UAM\demo\figures\communication_evolution.png)

建议三段口述：

1. 任务开始建立必要连接  
2. 分散阶段通信变稀疏  
3. 高风险/靠近目标时连接增强  

对照：GAT 持续密边。

---

# 11. 贡献总结

1. **Adaptive Communication DSGF**：拓扑 = 可学习决策变量  
2. **Budget-aware mechanism**：资源约束下保持任务能力  
3. **系统验证**：~91× Comm↓、Success 可比、Packet-loss 稳健、触发可解释  

# 

1. 将工作定位为 **“通信拓扑学习 / 带宽约束协同”**，而非“提高 Success 的 MARL”——是否同意？  
2. **16 UAV × 5 seeds + Budget + Packet Loss** 是否够 RA-L/TASE？还缺什么？  
3. 投稿前是否必须做 **真机/硬件在环**，还是可作为后续工作？

