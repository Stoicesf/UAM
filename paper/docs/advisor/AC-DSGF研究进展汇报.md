# 面向通信受限多无人机协同任务的自适应通信拓扑学习方法

## ——AC-DSGF (Adaptive Communication-aware DSGF)

**研究进展汇报**（导师简报 · 约 10 分钟）

> 定位：不是论文全文，而是说明**研究价值、创新点、实验可信度、投稿路线**，并征求指导意见。  
> 建议展示顺序：**框架图 → 主实验表 → Pareto → Packet Loss → Demo**

路径索引：
- 论文：`paper/ac_dsgf/AC_DSGF_v1.pdf`（7 页）
- 图：`paper/ac_dsgf/figures/`、`paper/figures/`
- Demo：`demo/videos/ac_dsgf_dynamic_comm.mp4`

---

# 1. 研究背景与问题定义

## 1.1 多无人机协同面临的核心挑战

无人机蜂群在搜索巡检、灾害救援、目标追踪等场景中需要共享位置、环境与任务信息。实际系统中通信却受限于：

- 带宽有限
- 能耗约束
- 通信延迟
- 网络不稳定

因此核心问题不是“策略是否更复杂”，而是：

> **无人机是否应该持续通信？应该和谁通信？通信多少？**

## 1.2 现有方法不足

| 路线 | 代表 | 问题 |
|------|------|------|
| 独立 RL | MAPPO | 缺少显式协同信息交换 |
| 图通信 MARL | GAT / Graph MARL | 拓扑多为半径/全连接等人设，难适应任务变化与预算 |

**研究缺口：** 多数工作优化“如何决策”，很少优化“如何通信”。

**研究问题：**

> 能否把**通信拓扑选择本身**作为强化学习中的可学习决策变量？

---

### 【插入图片 1】研究动机

| 项 | 内容 |
|----|------|
| 文件 | `paper/ac_dsgf/figures/fig_motivation_degradation.png`（或 `paper/figures/fig_motivation_degradation.png`） |
| 标题建议 | Motivation: guided MARL can collapse without residual decoupling; AC-DSGF keeps guidance assistive while optimizing communication |
| 展示方式 | PPT 第 2 页全幅；口头一句：“固定/过强通信引导会崩，我们既要稳策略，也要省通信” |

（补充视觉可用 Demo 中“GAT 密边 vs AC 稀疏边”截帧对照。）

---

# 2. 核心思想与创新点

## 2.1 核心思想

**AC-DSGF**：Adaptive Communication-aware Dynamic Sparse Graph Fusion

把通信拓扑从“人为设定”变为“智能体自主学习”——**通信边也是决策**。

传统管道：

```text
policy → (预定义) communication → environment
```

我们的转变：

```text
environment → learn communication topology → policy
```

**核心句（建议对导师强调）：**

> Unlike previous approaches that assume communication topology is predefined, we treat topology selection as a learnable decision process.

## 2.2 三点贡献（与论文 C1–C3 对齐）

1. **可学习通信拓扑（Gate）** \(g_{ij}\in[0,1]\)
2. **预算约束边选择（Top-K / Budget）**
3. **残差策略解耦** \(a=\pi(o)+\beta\Phi\)，通信辅助而非支配策略

主主张（务必避免说“Success 最高”）：

> **在通信约束下保持可比任务表现，同时大幅降低通信开销。**

---

# 3. 方法框架

```text
Observation
    ↓
Communication Gate  g_ij
    ↓
Adaptive Graph
    ↓
Budget Constraint (Top-K)
    ↓
Residual Policy  a = π(o) + βΦ
    ↓
Action
```

### 3.1 Adaptive Communication Gate

\[
g_{ij}=\sigma\!\big(W_g[h_i,h_j,d_{ij},q_{ij}]\big)\in[0,1]
\]

软门控（非二值边）；决策依赖状态、距离与交互风险。

### 3.2 Budget-aware Communication

在候选边上做 Top-K，形成 \(A^{AC}=A\odot E^K\)，保证通信预算受控。

### 3.3 Residual Policy Guidance

\[
a=\pi(o)+\beta\Phi
\]

避免通信模块“劫持”策略（消融：去掉残差后 Success 可崩溃约 42×，见表消融）。

---

### 【插入图片 2】整体框架（核心）

| 项 | 内容 |
|----|------|
| 文件 | `paper/ac_dsgf/figures/framework.png` |
| 标题建议 | AC-DSGF Framework: Gate → Budget → Residual Policy |
| 展示方式 | PPT 主视觉；讲解顺序 Gate → Budget → Residual |

---

# 4. 与已有方法区别

### 【插入表 1】方法对比

| 方法 | 通信方式 | 拓扑是否学习 | 预算约束 |
|------|----------|--------------|----------|
| MAPPO | 无显式图通信 | × | × |
| GAT | 固定/半径图 | × | × |
| DSGF | 质量加权动态稀疏图 | 部分（质量） | ×（默认用满半径支撑） |
| **AC-DSGF** | 动态学习门控拓扑 | **√** | **√** |

区别不是“网络更复杂”，而是：**学习通信行为**，并把拓扑当作决策变量。

---

# 5. 实验设计（可信度要点）

| 参数 | 设置 |
|------|------|
| 任务 | Multi-UAV Navigation（VMAS） |
| 规模 | **16 UAV** 主表；消融 4 UAV（**勿混比**） |
| 方法 | MAPPO / GAT / Transformer / DSGF / AC-DSGF |
| 训练 | 102k frames；主表 **5 seeds**；eval **200 ep** |
| 指标 | Success / Collision / Comm / CEI |
| 辅助 | Budget sweep、Silence collapse、Trigger、Gate stability、Packet loss、Inference cost |

**审稿友好表述：** Demo ≠ 主表；Budget/Packet-loss 部分为诊断协议（64 ep），趋势有效，绝对值勿与 Table I 混谈。

---

# 6. 实验结果 1：总体性能（主结果）

### 【插入表 2】Table I（16 UAV · 5 seeds）

| Method | Success (%) | Comm | CEI |
|--------|-------------|------|-----|
| MAPPO | 2.40±0.90 | 0 | — |
| GAT | 2.04±0.90 | 38.80 | 0.0005 |
| Transformer | 0.70±0.16 | 240 | ≈0 |
| DSGF | **4.01±1.68** | 39.27 | 0.0010 |
| **AC-DSGF** | **3.95±0.83** | **0.43** | **0.0913** |

**口述要点：** Success 与 DSGF **可比**；通信 **39.27 → 0.43（约 90×）**；卖点是通信效率，不是刷榜 Success。

CSV：`paper/tables/table1_final.csv`

---

### 【插入图片 3】通信效率 Pareto

| 项 | 内容 |
|----|------|
| 文件 | `paper/ac_dsgf/figures/fig_comm_pareto.png` |
| 说明 | X=Comm（log），Y=Success；AC-DSGF 在高效率区（可比 S、低 C） |
| 展示方式 | 与表 2 同页；导师一眼看到“低通信点” |

---

# 7. 实验结果 2：通信预算（Budget）

验证：收紧预算时任务是否塌方。

| Budget | GAT | DSGF | AC-DSGF |
|--------|-----|------|---------|
| 100% | 21.3 | 21.7 | **27.1** |
| 50% | 18.2 | 21.6 | **25.2** |
| 10% | 16.9 | 21.7 | **24.2** |

结论：AC 在预算大幅下降时仍保持任务表现（*graceful degradation*）。  
（诊断协议，趋势用于论证，勿与 Table I 混比绝对值。）

### 【插入图片 4】

| 文件 | `fig_comm_budget16.png`（`paper/figures/` 或 `paper/ac_dsgf/figures/`） |
| 说明 | 横轴 Budget，纵轴 Success |

---

# 8. 实验结果 3：通信可靠性（Packet Loss）

现实通信会丢包。固定 16 UAV 模型，边随机丢失 0%–70%。

| Loss | GAT | AC-DSGF |
|------|-----|---------|
| 0% | 21.2 | 26.2 |
| 70% | 15.8 | **25.8** |

GAT 相对约下降 25%；AC-DSGF 几乎持平 → **不可靠链路上的 graceful degradation**。

### 【插入图片 5】

| 文件 | `fig_packet_loss.png` |
| 标题建议 | Performance under unreliable communication |

---

# 9. 通信行为可解释性（不是随机关通信）

| 因素 | 与 Comm 相关 |
|------|----------------|
| 邻近风险 nn_risk | **+0.84** |
| 空间分散 | −0.57 |
| 目标距离 | ≈0 |

Gate stability：\(\mathrm{Var}(C)\approx 0.003\)，**>99%** 时间步 \(\Delta E=0\) → 动态但非抖动乱切。

### 【插入图片 6】

| 文件 | `fig_comm_trigger.png`；辅图 `fig_gate_stability.png` |

---

# 10. Demo 展示

**目的：** 展示“学会动态调整通信”，不是证明成功率第一。

| 项 | 路径 |
|----|------|
| 视频 | `demo/videos/ac_dsgf_dynamic_comm.mp4` |
| 曲线 | `demo/figures/communication_evolution.png` |

建议播放 ~30 s：对比 GAT 密边 vs AC 稀疏/任务触发边。

**【展示方式】** 汇报末尾播放；口头：“这是定性补充，正式结论以 Table I + Packet Loss 为准。”

---

# 11. 当前论文与工程状态

| 模块 | 状态 |
|------|------|
| 算法（DSGF v2 + AC） | 冻结完成 |
| 主表 16UAV×5 seeds | 完成 |
| 消融 / Budget / Trigger / Silence / Packet loss / Compute | 完成 |
| IEEE 初稿 PDF | **`AC_DSGF_v1.pdf`（7 pages）** |
| Cover letter / Mock R1–R3 答复骨架 | 已备 |

投稿定位（自我评估）：**RA-L / TASE / TIV 方向更匹配**（工程通信约束）；T-RO 可能还需更大规模/真机。

---

# 12. 下一步计划

**短期**
1. 按导师意见改论文叙事与图表  
2. 完成 Supplement（超参 / 网络结构）  
3. 投稿前润色（降 AI 味、引用核对）

**中长期**
- 真机 / 半实物通信协议  
- 更大规模 swarm  
- 与具体无线电模型对齐的 packet 定义  

---

# 13. 请导师指导的问题（建议原话）

1. 将论文定位为 **“通信拓扑学习 / 带宽约束下的协同”**，而不是“单纯提高 MARL Success”——是否合理？  
2. 当前 **16 UAV · 5 seeds + Packet loss + Budget** 的实验规模，是否达到您期望的 **IEEE RA-L / TASE** 投稿门槛？还差什么？  
3. 是否需要在投稿前增加 **真机或硬件在环** 实验，还是可作为后续工作？

---

# 汇报收束（一句话）

> **本研究不是设计一个更复杂的多智能体策略网络，而是让无人机自主学习通信拓扑，在有限通信资源下保持任务完成能力，实现通信效率与协同性之间的平衡。**

---

## 附录：材料速查

| 类型 | 路径 |
|------|------|
| 论文 PDF | `paper/ac_dsgf/AC_DSGF_v1.pdf` |
| 投稿包 | `paper/ac_dsgf_final/` |
| 主表 | `paper/tables/table1_final.csv` |
| Packet loss | `paper/tables/table_packet_loss.csv` |
| 框架图 | `paper/ac_dsgf/figures/framework.png` |
| Pareto | `paper/ac_dsgf/figures/fig_comm_pareto.png` |
| Demo | `demo/videos/ac_dsgf_dynamic_comm.mp4` |
| Mock 答复 | `paper/ac_dsgf/response_mock.md` |
