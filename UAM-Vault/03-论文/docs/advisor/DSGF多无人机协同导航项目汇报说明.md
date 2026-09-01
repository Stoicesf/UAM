# DSGF：面向多无人机协同导航的动态图时空融合强化学习方法



> 
> 
> **叙事主线（务必统一口径）：**  
> 我们提出 DSGF，针对动态图协同中的**长期任务退化**问题。  
> **正式实验验证算法有效性；Demo 只展示算法行为合理性。二者职责分开，不可互相替代。**

| 项目 | 内容 |
|------|------|
| 算法状态 | DSGF v2 **已冻结**（不改结构、不调 λ） |
| 正式证据 | Table I–IV，见 `paper/experiment_freeze.json` |
| Demo | `demo/videos/`（定性展示） |
| 相关写作 | `paper/chapter3_method.tex`、`paper/chapter4_exp.tex` |

---

## 文档阅读说明：图片标注约定

下文中：

```
【📷 图片｜标题】
路径: ...
作用: ...
讲解要点: ...
```

表示该位置应插入对应图片。  
相对路径默认以仓库根目录 `UAM/` 为基准。

---

# 一、项目背景与研究意义

随着无人机技术的发展，多无人机系统（Multi-UAV System）在搜索救援、环境监测、目标跟踪、区域巡检等任务中具有重要应用价值。

相比单无人机系统，多无人机蜂群能够通过空间分布和信息共享提高任务覆盖范围与执行效率。然而，在实际复杂环境中，多无人机协同仍然面临以下挑战。

### 1. 局部观测限制

每个无人机只能获取自身附近环境信息 \(o_i^t\)，无法直接获得全局状态，需要通过协同机制完成任务决策。

### 2. 动态通信约束

无人机之间通信关系随位置变化不断改变：

\[
G_t = (V, E_t)
\]

固定通信拓扑难以适应动态任务环境。

### 3. 长期训练稳定性问题

现有图强化学习方法能够提高短期协同能力，但常出现：

- 过度依赖邻居信息  
- 策略耦合增强  
- **长期任务成功率下降**

因此，本项目关注：

> **如何在动态通信环境下，使多无人机系统获得长期更稳定的协同行为。**

---

【📷 图片｜多无人机协同问题场景】

- **路径：** `paper/figures/fig_pptx_problem_scene.png`
- ![fig_pptx_problem_scene](F:\UAM\paper\figures\fig_pptx_problem_scene.png)
- **作用：** 建立问题对象——局部观测圈、有限通信边、共享目标。
- **讲解要点：**「每个 UAV 只看见虚线圆内的局部信息；连线是受限通信，不是全连接全球信息。」

---

# 二、研究问题定义

本项目研究多智能体协同导航任务：给定 \(N\) 个无人机智能体、局部观测、动态通信网络与共享目标，学习去中心化策略

\[
\pi_i(a_i \mid o_i)
\]

使系统最大化任务收益（成功到达、低碰撞、可控通信开销）。

### 环境与框架

| 参数 | 设置 |
|------|------|
| 仿真平台 | VMAS Multi-Agent Navigation |
| Agent 规模 | 4 / 8 / 16 / 32 UAV |
| 算法框架 | Multi-Agent RL（CTDE） |
| 基础算法 | MAPPO |
| 指导策略 | GAT / Transformer / **DSGF v2** |
| 评价指标 | Success、Collision、Reward、Communication Cost、AULC |

**重要设定区分（写作与口头汇报均不得混用）：**

| 用途 | 规模 | 说明 |
|------|------|------|
| Table I 主对比 | 16 UAV · 5 seeds | 正式性能证据 |
| Table II 消融 | 4 UAV · seed=42 | 机制证据 |
| Demo | 4 UAV · 可视化选优 | **行为展示，非统计证明** |

---

# 三、现有方法分析与研究动机

## 3.1 MAPPO

MAPPO 采用集中训练、分散执行（CTDE）：训练期可利用联合信息，执行期各智能体仅用本地策略 \(\pi_i(a_i \mid o_i)\)。

- **优势：** 训练相对稳定、实现成熟  
- **不足：** 缺少显式通信 / 邻居结构，难以系统性利用交互关系  

## 3.2 GAT 图强化学习

GAT 通过图注意力聚合邻居信息：

\[
h_i = \sum_j \alpha_{ij} h_j
\]

- **优势：** 能建模通信关系，短期协同常有提升  
- **不足：** 半径内邻居易被一视同仁；引导过强时易 **over-coupling**  

## 3.3 关键动机：长期训练退化

本项目早期实验观察到（4 UAV 图引导设定）：

| 方法 | 20k Success | 102k Success |
|------|------------:|-------------:|
| GAT | 4.74% | 0.22% |

同期训练 reward 仍可上升，但评估成功率下降。这说明：

> **Reward 优化并不能自动保证长期任务泛化。**

这一现象构成 DSGF 设计的直接动机：既要利用图信息，又要避免引导信号主导策略、造成长期退化。

---

【📷 图片｜MAPPO / GAT / DSGF 结构对比】

- **路径：** `paper/figures/fig_pptx_method_compare.png`
- ![fig_pptx_method_compare](F:\UAM\paper\figures\fig_pptx_method_compare.png)
- **作用：** 从「无显式通信 → 固定/半径图 → 动态稀疏融合」说明为何需要新方法。

【📷 图片｜Motivation：GAT 长期退化】

- **路径：** `paper/figures/fig_pptx_motivation_degradation.png`
- ![fig_pptx_motivation_degradation](F:\UAM\paper\figures\fig_pptx_motivation_degradation.png)
- **作用：** 把「问题不是没做图，而是长期退化」讲清楚；这是科研动机页，优先级极高。
- **讲解要点：**「短期变好、长期变差 —— 所以我们不是再堆一个更大的图网络，而是改耦合方式。」

---

# 四、DSGF 方法设计

DSGF（Dynamic Sparse Graph Fusion）核心思想：

> **不让无人机盲目依赖通信信息，而是动态选择有效邻居，并以残差方式辅助策略决策。**

整体流程：

```
Local Observation
        ↓
Dynamic Graph Construction   Ã = A · q
        ↓
Sparse Spatial Attention
        ↓
Temporal Feature Fusion (GRU)
        ↓
Residual Policy   a = π(o) + β·Δa(Φ)
        ↓
Action
```

训练仍遵循 CTDE：Actor 去中心化，Critic 集中估计价值。DSGF v2 **关闭** guidance alignment 奖励，以任务回报为主优化信号（配置冻结于 `configs/dsgf/dsgf_v2_frozen.yaml`）。

---

【📷 图片｜DSGF 整体框架（汇报核心图）】

- **路径：** `paper/figures/fig_pptx_dsgf_pipeline.png`
- ![fig_pptx_dsgf_pipeline](F:\UAM\paper\figures\fig_pptx_dsgf_pipeline.png)
- **作用：** 让听众把视频里的「飞机在飞」映射到模块链路；**比 Demo 更重要的概念图。**
- **讲解要点：**「关键不是最后出来一个动作，而是动作如何被动态图与残差约束生成。」

---

# 五、核心创新模块

## 5.1 Dynamic Graph（动态通信图）

传统二值邻接：

\[
A_{ij} = \mathbf{1}\!\left[\|p_i - p_j\| < R_c\right]
\]

DSGF 引入通信质量 \(q_{ij}\)：

\[
\tilde{A}_{ij} = A_{ij}\, q_{ij}
\]

\(q_{ij}\) 综合距离、相对运动 / 对齐等信息，降低几何可行但信息量低的冗余边。  
消息传递复杂度在平均度数 \(k \ll N\) 时接近 \(O(Nkd)\)（见 Table V）。

---

【📷 图片｜动态稀疏图可视化】

- **路径：** `demo/figures/graph_attention_dsgf.png`
- ![graph_attention_dsgf](F:\UAM\demo\figures\graph_attention_dsgf.png)
- **可选对比：** `demo/figures/graph_attention_gat.png`
- **作用：** 展示边非永恒全连接；拓扑随时间变化。
- **讲解要点：**「边在变 —— 这就是 Dynamic Sparse Graph，而不是画一个静态组织架构图。」

## 5.2 Spatiotemporal Fusion（时空融合）

- **Spatial Attention：** 在 \(\tilde{A}\) 支撑上聚合「谁值得听」  
- **Temporal Memory（GRU）：** 保留短时历史，减轻单帧近视  

## 5.3 Residual Policy（核心创新）

若直接 \(a \approx \pi(o,\Phi)\) 且 \(\Phi\) 过强，易出现 over-guidance。  
DSGF：

\[
a_i = \pi_\theta(o_i) + \beta(t)\, \Delta_\phi(\Phi_i)
\]

- 主策略 \(\pi(o)\) 保留自主  
- \(\beta(t)\) 退火：早期借协同，后期减弱过度依赖  

**机制证据（4 UAV 消融，勿与 Table I 混用）：**

| Variant | Success |
|---------|--------:|
| w/o Dynamic Graph | 0.28% |
| w/o Temporal | 0.86% |
| **w/o Residual** | **0.22%** |
| **Full DSGF** | **9.27%** |

去掉 Residual 后性能崩溃约 **42×**，表明残差解耦对准了「长期过依赖」这一失败模式。

---

【📷 图片｜Residual 机制示意】

- **路径：** `paper/figures/fig_pptx_residual.png`
- ![fig_pptx_residual](F:\UAM\paper\figures\fig_pptx_residual.png)
- **作用：** 对比「Guide 主导」与「本地策略 + 可衰减修正」。

【📷 图片｜消融结果】

- **路径：** `paper/figures/fig4_ablation.png`
- ![fig4_ablation](F:\UAM\paper\figures\fig4_ablation.png)
- **作用：** 用实验数字支撑 Residual 不是「多一个模块」，而是关键机制。

---

# 六、实验设计与正式结果

> 本节是**有效性证明**。Demo 不得用来替代本节数字。

为验证 DSGF，项目设计五类正式实验（算法冻结后完成）。

## 6.1 Baseline 比较（Table I）

- **设定：** 16 UAV，5 seeds，102k frames  
- **方法：** MAPPO / GAT / Transformer / DSGF  

| Method | Success (mean±std) |
|--------|-------------------:|
| MAPPO | 2.40% ± 0.90% |
| GAT | 2.04% ± 0.90% |
| Transformer | 0.70% ± 0.16% |
| **DSGF** | **4.01% ± 1.68%** |

**严谨表述：**

- DSGF 取得最高**平均**成功率（相对 MAPPO / GAT 有提升）。  
- 绝对成功率仍低（大规模连续协同极难）。  
- 相对 MAPPO 的显著性检验未稳定达到 \(p<0.05\)（t-test ≈0.11，Wilcoxon ≈0.06），汇报中说「一致数值优势 / 需更多 seeds 加强统计」比说「显著碾压」更安全。

---

【📷 图片｜Baseline Success 柱状图】

- **路径：** `paper/figures/fig1_success_5seed.png`
- ![fig1_success_5seed](F:\UAM\paper\figures\fig1_success_5seed.png)
- **可选：** `paper/figures/fig2_learning_curve_5seed.png`（学习过程）
- ![fig2_learning_curve_5seed](F:\UAM\paper\figures\fig2_learning_curve_5seed.png)

## 6.2 消融实验（Table II）

见第五节表。结论：**Residual 是长期稳定性的关键。**

## 6.3 扩展性（Scalability）

测试 \(N \in \{4,8,16,32\}\)。DSGF 在冻结配方下可完成大规模训练与评估协议；绝对成功率随规模上升而变难，报告中强调「可扩展实验协议 + 难度刻画」，而非饱和性能。

---

【📷 图片｜可扩展性曲线】

- **路径：** `paper/figures/fig3_scalability.png`
- ![fig3_scalability](F:\UAM\paper\figures\fig3_scalability.png)

## 6.4 通信效率（Table IV）

改变通信半径 \(R \in \{1,2,\mathrm{full}\}\)（16 UAV）。

| | R=1 | R=2 | full |
|--|----:|----:|-----:|
| GAT | 1.88% | 1.28% | **3.19%** |
| DSGF | **2.22%** | **4.07%** | 2.41% |

**安全叙事：**

> DSGF 在**受限通信预算**下保持更有竞争力的成功率；  
> **不声称**在任意连通条件下全面优于 GAT。

---

【📷 图片｜通信 Pareto】

- **路径：** `paper/figures/fig6_communication.png`
- ![fig6_communication](F:\UAM\paper\figures\fig6_communication.png)

## 6.5 泛化（Table III）

在障碍密度上训练后，对未见过的障碍数量做 zero-shot 评估，检验环境偏移下的行为稳健性（详见 `paper/tables/table3_generalization.csv`）。

---

【📷 图片｜障碍泛化】

- **路径：** `paper/figures/fig5_generalization.png`
- ![fig5_generalization](F:\UAM\paper\figures\fig5_generalization.png)

## 6.6 复杂度（Table V）

| Method | Complexity |
|--------|------------|
| MAPPO | \(O(Nd)\) |
| GAT / Transformer | \(O(N^2 d)\) |
| DSGF | \(O(Nkd + Nd)\)（\(k \ll N\) 时近线性） |

路径：`paper/tables/table5_complexity.tex`

---

# 七、Demo 说明（定性展示，非统计证明）

## 7.1 Demo 目的（必须先讲清）

| | 正式实验 | Demo |
|--|----------|------|
| 问题 | 算法是否有效？ | 算法行为是否合理、可解释？ |
| 证据 | Table I–IV、多 seed / 扫参 | 轨迹、通信边、对比视频 |
| 口径 | mean±std | **best episode 可视化** |

**推荐口径：**

> Demo 用于直观展示 DSGF 学到的多无人机协同行为，包括动态通信、路径协调与任务执行过程。

---

## 7.2 Demo 环境设置

| 项目 | 值 |
|------|----|
| 环境 | VMAS Cooperative Navigation（plain，无障碍） |
| UAV 数量 | 4 |
| 算法 | DSGF v2 / MAPPO / GAT-A |
| Checkpoint | **20k early checkpoint**（见下节） |
| 展示方式 | 多 episode 中按 **best episode** 选优可视化 |
| Demo 选优 Success | MAPPO 50% / GAT 50% / DSGF **75%**（仅展示，非均值） |

---

【📷 图片｜Demo Environment】

- **路径：** `paper/figures/fig_pptx_demo_environment.png`
- ![fig_pptx_demo_environment](F:\UAM\paper\figures\fig_pptx_demo_environment.png)
- **必须口头强调：** best episode，不是 average success 75%。

## 7.3 为什么选择 checkpoint_20k？

项目观察到训练后期可能出现 **reward 上升、task success 下降** 的背离。  
正式表格报告完整训练与多 seed 统计；**可视化 Demo 选取 task-success 更优的早期检查点**，以展示策略行为，而非伪装成更高的平均指标。

可写：

> 由于长期训练存在一定任务泛化退化现象，Demo 采用性能峰值附近的 checkpoint 展示策略行为。

这体现科研严谨，而不是「挑好看结果作弊」——前提是汇报者主动说明。

---

【📷 图片｜Checkpoint Selection】

- **路径：** `paper/figures/fig_pptx_checkpoint_selection.png`
- ![fig_pptx_checkpoint_selection](F:\UAM\paper\figures\fig_pptx_checkpoint_selection.png)
- **作用：** 标明 20k = Demo checkpoint；与 102k 正式评估分流。
- **可选辅图：** 相关学习曲线 `paper/figures/fig2_learning_curve_5seed.png`

## 7.4 Demo 展示内容

### （1）定性对比（优先播放）

| Method | 期望观察的行为（定性） |
|--------|------------------------|
| MAPPO | 更偏独立探索，协调偏弱 |
| GAT | 交互更强，但易过耦合 / 行为板结 |
| DSGF | 自适应协调，通信边更「按需」 |

---

【📷 图片｜三算法轨迹对比】

- **路径：** `demo/figures/comparison_trajectories.png`
- ![comparison_trajectories](F:\UAM\demo\figures\comparison_trajectories.png)
- **分图：** `trajectory_mappo.png` / `trajectory_gat.png` / `trajectory_dsgf.png`

【🎬 视频｜并排对比】

- **路径：** `demo/videos/comparison_mappo_gat_dsgf.mp4`
- **作用：** 定性对照；旁白对齐「行为」，不要报成功率战报。

### （2）DSGF 行为过程拆解（推荐）

| 阶段 | 看什么 | 建议说明 |
|------|--------|----------|
| Stage 1 Observation | 初始散布 + 建图 | 局部观测 → 动态图 |
| Stage 2 Coordination | 中段轨迹 | 稀疏聚合邻居信息、避让与协同 |
| Stage 3 Completion | 接近目标 | 形成协同导航结果 |

---

【📷 图片｜三阶段行为拆解】

- **路径：** `demo/figures/dsgf_behavior_stages.png`
- ![dsgf_behavior_stages](F:\UAM\demo\figures\dsgf_behavior_stages.png)
- **单方法视频：** `demo/videos/dsgf_4uav.mp4`
- 
- <video src="F:\UAM\demo\videos\dsgf_4uav.mp4"></video>

### （3）动态通信

配合 `graph_attention_dsgf.png` 说明：边随时间增减，不是固定通信模板。

---

# 八、项目贡献总结（Key Contributions）

### 贡献 1：Dynamic Communication Modeling

将静态 / 纯几何图推进为**通信质量感知的动态稀疏图** \(\tilde{A}=A\cdot q\)，在受限半径下选择更有效邻居。

### 贡献 2：Long-horizon Stability via Residual Policy

正视图引导的 **短期收益—长期退化** 矛盾，提出残差解耦策略 \(a=\pi(o)+\beta\Delta a(\Phi)\)，消融表明 Residual 为关键模块。

### 贡献 3：Extensive Evaluation Protocol

形成可复现证据链：Baseline（5-seed）· Ablation · Scalability · Communication Pareto · Zero-shot Generalization · Complexity · **定性 Demo**。

---

# 九、项目当前状态

### 已完成

| 类别 | 状态 |
|------|------|
| MAPPO / GAT / Transformer baseline | ✅ |
| DSGF v1 → v2（冻结） | ✅ |
| 16 UAV × 5 seeds（Table I） | ✅ |
| 消融（Table II） | ✅ |
| 泛化（Table III） | ✅ |
| 通信效率（Table IV）+ Pareto 图 | ✅ |
| 复杂度表（Table V） | ✅ |
| Scalability 4–32 | ✅ |
| 论文级 Demo（含三算法对比） | ✅ |
| Method / Exp LaTeX 草稿 | ✅ |

### 当前阶段（双轨）

| Track | 内容 | 原则 |
|-------|------|------|
| **A · DSGF IEEE** | 写作：Related Work / Abstract / `main.tex` | **算法冻结，不再调参** |
| **B · AC-DSGF** | 下一代自适应通信框架 | 详见 [`ac_dsgf_roadmap.md`](./ac_dsgf_roadmap.md) |

AC-DSGF **复用**全部 DSGF 实验作强 baseline，只新增：通信决策层 + 预算/故障实验 + 新 Demo。骨架已就位：`models/communication/`、`models/ac_dsgf.py`。

---

# 十、汇报时建议讲述顺序

```
1. Background          为何需要蜂群协同
2. Problem             局部观测 + 动态通信约束
3. Existing Limitation MAPPO / GAT 不足
4. Motivation          长期训练退化（关键）
5. DSGF Method         框架图（最核心）
6. Experimental Results Table I / II / IV
7. Demo                行为展示（明确非统计证明）
8. Contributions       三点贡献
9. Q&A
```



---

# 十一、Q&A 预备

| 可能问题 | 建议回答 |
|----------|----------|
| Success 绝对值为什么这么低？ | 16 UAV 连续协同很难；贡献是相对提升与机制证据，不是宣称已可实飞部署。 |
| Demo 75% 和论文 4% 矛盾吗？ | 不矛盾：前者是 4 UAV、early ckpt、best episode 可视化；后者是 16 UAV、5-seed 均值。 |
| 统计显著性不够？ | 诚实承认 seed 预算有限；报告均值优势，并说明扩大 seed 是写作阶段后续工作之一。 |
| full 通信 GAT 更高？ | 优势主要在受限半径；叙事是通信预算下的稳健性，不是无条件最优。 |
| Ablation 与 Table I 数字差很多？ | 4 UAV 与 16 UAV 设定不同，机制表与主表不可混比绝对值。 |

---

# 附录 A：图片与视频总索引

| 编号 | 内容 | 路径 | 用于章节 |
|------|------|------|----------|
| A1 | 问题场景 | `paper/figures/fig_pptx_problem_scene.png` | 一 |
| A2 | 方法对比 | `paper/figures/fig_pptx_method_compare.png` | 三 |
| A3 | 退化动机 | `paper/figures/fig_pptx_motivation_degradation.png` | 三 |
| A4 | DSGF 框架 | `paper/figures/fig_pptx_dsgf_pipeline.png` | 四 |
| A5 | Residual 示意 | `paper/figures/fig_pptx_residual.png` | 五 |
| A6 | 动态图 | `demo/figures/graph_attention_dsgf.png` | 五 / 七 |
| A7 | Table I 柱状图 | `paper/figures/fig1_success_5seed.png` | 六 |
| A8 | Ablation | `paper/figures/fig4_ablation.png` | 五 / 六 |
| A9 | Scalability | `paper/figures/fig3_scalability.png` | 六 |
| A10 | Comm Pareto | `paper/figures/fig6_communication.png` | 六 |
| A11 | Generalization | `paper/figures/fig5_generalization.png` | 六 |
| A12 | Demo 环境 | `paper/figures/fig_pptx_demo_environment.png` | 七 |
| A13 | Checkpoint 选择 | `paper/figures/fig_pptx_checkpoint_selection.png` | 七 |
| A14 | 三算法轨迹 | `demo/figures/comparison_trajectories.png` | 七 |
| A15 | 行为三阶段 | `demo/figures/dsgf_behavior_stages.png` | 七 |
| V1 | 对比视频 | `demo/videos/comparison_mappo_gat_dsgf.mp4` | 七 |
| V2 | DSGF 单视频 | `demo/videos/dsgf_4uav.mp4` | 七 |

# 

|      |      |
|------|------|
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |


