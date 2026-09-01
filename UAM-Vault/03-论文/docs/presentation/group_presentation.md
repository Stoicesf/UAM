# DSGF：面向多无人机协同导航的动态图时空融合强化学习方法

> **文档角色：** PPT 制作提纲 / 页级讲稿脚本  
> **组员与老师提前阅读，请用正式技术说明：**  
> [`DSGF多无人机协同导航项目汇报说明.md`](./DSGF多无人机协同导航项目汇报说明.md)  
>  
> **主线一句话：**  
> 我们提出 DSGF，针对动态图协同中的**长期任务退化**问题；正式实验验证有效性，Demo 只用于展示机制产生的协同行为。

**汇报人：** XXX  
**算法状态：** DSGF v2 已冻结  
**正式证据：** Table I–IV（`paper/experiment_freeze.json`）  
**定性展示：** `demo/videos/comparison_mappo_gat_dsgf.mp4`

---

## 推荐汇报顺序（不要一上来播 Demo）

| 页 | 内容 | 图片 / 素材 |
|----|------|-------------|
| 1 | 标题 | — |
| 2 | Background：为何需要蜂群协同 | `fig_pptx_problem_scene.png` |
| 3 | Problem & Limitation：MAPPO / GAT | `fig_pptx_method_compare.png` |
| 4 | **Motivation**：长期训练退化 | `fig_pptx_motivation_degradation.png` ⭐ |
| 5 | DSGF Method 框架 | `fig_pptx_dsgf_pipeline.png` ⭐ |
| 6 | Innovation：动态图 + Residual | `graph_attention_dsgf.png` + `fig_pptx_residual.png` |
| 7 | Experimental Results（Table） | Fig1 / Fig4 / Fig6 |
| 8 | Demo Environment + Checkpoint 选择 | `fig_pptx_demo_environment.png` + `fig_pptx_checkpoint_selection.png` |
| 9 | Qualitative Comparison | `comparison_trajectories.png` + 视频 |
| 10 | Behavior Process（三阶段） | `dsgf_behavior_stages.png` |
| 11 | Key Contributions | — |
| 12 | Q&A | — |

---

# 1. Background：为什么需要多无人机协同

搜索救援、环境监测、目标追踪等任务中，单机覆盖能力有限，需要多 UAV **去中心化协同决策**。

每个 UAV 只持有局部观测：

```
oᵢ → πᵢ → aᵢ
```

同时受通信半径、带宽与拓扑动态变化约束。

---

### 【图片｜问题场景】

`paper/figures/fig_pptx_problem_scene.png`

![fig_pptx_problem_scene](F:\UAM\paper\figures\fig_pptx_problem_scene.png)

**口播：** 局部观测圈 + 有限通信边 + 共同目标 —— 这是问题设定，不是我们的贡献。

---

# 2. Problem：现有方法的不足

## MAPPO

- CTDE，训练稳定，**无显式通信图**
- 难以编码邻居结构化交互 → 协同偏弱

## GAT-MAPPO

- 引入半径通信图，短期可改善协调
- 但邻居近似一视同仁；**长期训练易出现任务退化**

---

### 【图片｜方法对比】

`paper/figures/fig_pptx_method_compare.png`

---

# 3. Motivation：长期训练退化（科研核心切入点）

**关键观察（4-UAV 试点，写进论文 Motivation）：**

| Method | ~20k Success | ~102k Success |
|--------|-------------:|--------------:|
| GAT-MAPPO | ≈ 4.74% | ≈ 0.22% |
| DSGF v2 | ≈ 26.27% | ≈ 9.27% |

训练回报仍可上升，但 **eval success 下降** ——  
说明：优化 surrogate / 图引导并不保证任务泛化（over-specialization）。

> 这不是“少训一会更好”的工程 trick，而是要解决的科学问题：  
> **如何在通信引导下保持长期任务稳定性？**

DSGF 的 Residual Policy 正是对这一问题的直接回应。

---

### 【图片｜Motivation 实验图】⭐ 比 Demo 更重要

`paper/figures/fig_pptx_motivation_degradation.png`

**口播：**「先看这个：图方法短期有用，长期 success 塌；我们要解决的是这件事。」

---

# 4. Method：DSGF 如何解决

## 核心思想

> 不应固定依赖全邻居通信；应按状态动态选择有效信息，并以**残差方式**注入策略，保留自主决策。

## 流水线

```
Local Observation
       ↓
Dynamic Graph Construction   Ã = A · q
       ↓
Sparse Spatial Attention
       ↓
Temporal Memory (GRU)
       ↓
Residual Policy   a = π(o) + β·Δa(Φ)
       ↓
Action
```

---

### 【图片｜DSGF Framework】⭐ 汇报最核心的结构图

`paper/figures/fig_pptx_dsgf_pipeline.png`

**口播：**「视频里无人机在跑，但创新在这条链路；没有这张图，Demo 只是动画。」

---

## Innovation 1 — Dynamic Communication Modeling

静态半径图 → **质量感知动态稀疏图** `Ã_ij = A_ij · q_ij`

## Innovation 2 — Residual Decoupling（稳住长期）

\[
a = \pi(o) + \beta(t)\,\Delta a(\Phi)
\]

Ablation（4 UAV，勿与 16 UAV Table I 混用）：

| Variant | Success |
|---------|--------:|
| w/o Residual | 0.22% |
| Full DSGF | **9.27%** |

---

### 【图片】

- 动态图：`demo/figures/graph_attention_dsgf.png`
- Residual：`paper/figures/fig_pptx_residual.png`

---

# 5. Experimental Results：用实验验证有效性

> 下面数字来自**冻结正式实验**，不是 Demo。

## 5.1 Main Comparison（Table I · 16 UAV · 5-seed）

| Method | Success mean±std |
|--------|-----------------:|
| MAPPO | 2.40% ± 0.90% |
| GAT | 2.04% ± 0.90% |
| Transformer | 0.70% ± 0.16% |
| **DSGF** | **4.01% ± 1.68%** |

**安全表述：**

- 有一致的相对提升（约 +1.6pp vs MAPPO）
- 绝对 success 仍低；设定很难
- 5-seed 显著性未达 0.05（p≈0.06–0.11）→ **不宣称统计碾压**

图：`paper/figures/fig1_success_5seed.png`

## 5.2 Ablation（Table II · 4 UAV）

Residual 去除导致约 **42×** 性能塌陷 → 机制证据。

图：`paper/figures/fig4_ablation.png`

## 5.3 Communication（Table IV · 16 UAV）

受限半径下 DSGF 更具竞争力（R=1/2）；**full 下不声称全面碾压 GAT**。

> DSGF maintains competitive success under restricted communication budgets.

图：`paper/figures/fig6_communication.png`

（可选）Complexity：`paper/tables/table5_complexity.tex`

---

# 6. Demo：展示行为合理性（≠ 效果证明）

## 角色分工（必须讲清）

| | 作用 |
|--|------|
| **正式实验** | 验证算法是否有效（Table / Figure） |
| **Demo** | 直观展示协同行为是否合理（机制可读） |

❌ 不要说：  
「DSGF 显著提升成功率，所以做了 Demo。」

✅ 要说：  
「正式实验已给出量化证据；Demo 用来展示 DSGF 学到的动态通信与路径协调过程。」

---

## 6.1 Demo Environment

**图：** `paper/figures/fig_pptx_demo_environment.png`

```
Environment
- UAVs: 4
- Scenario: VMAS Cooperative Navigation（本 Demo 无障碍）
- Algorithm: DSGF v2（冻结）
- Checkpoint: 20k early-stop（task-success 峰值，非 final reward）
- Visualization: best episode among 8 rollouts（仅作示意）
```

⚠️ **禁止写成 “average success = 75%”**  
若提到单次 rollout：务必加限定词 **best episode / for illustration only**，并强调与 Table I（16 UAV 均值）不可比。

---

## 6.2 Checkpoint Selection（体现严谨）

**图：** `paper/figures/fig_pptx_checkpoint_selection.png`  
（Motivation 同页也可：`fig_pptx_motivation_degradation.png`）

**口播模板：**

> 由于图引导存在长期任务泛化退化，Demo **按任务成功率峰值 checkpoint（20k）** 可视化策略行为，  
> 而不是用最终训练步的权重。这是 RL 可视化常见做法，与 Table 中完整训练统计分开报告。

---

## 6.3 Qualitative Comparison（必须三算法并排）

**不要只播 DSGF。**

| Method | 定性行为（观察用语，非定量结论） |
|--------|----------------------------------|
| MAPPO | 偏独立探索，协同结构弱 |
| GAT | 交互更强，但易显“绑死/耦合” |
| DSGF | 自适应协调，通信边更稀疏动态 |

**静态图：** `demo/figures/comparison_trajectories.png`  
**视频：** `demo/videos/comparison_mappo_gat_dsgf.mp4`

**讲解看什么：** 轨迹冲突、是否趋同、通信边是否随时间变化 —— **不要报 Demo success 当论文结果。**

---

## 6.4 Behavior Process：拆成三阶段（视频旁贴图）

**组合图：** `demo/figures/dsgf_behavior_stages.png`

| Stage | 单帧 | 说明 |
|-------|------|------|
| 1 Observation | `demo/figures/stages/stage1_observation.png` | 局部观测；初始化动态通信图 |
| 2 Coordination | `demo/figures/stages/stage2_coordination.png` | 稀疏聚合邻居信息；路径开始协调 |
| 3 Completion | `demo/figures/stages/stage3_completion.png` | 接近目标；完成协同导航过程 |

**口播：**「不是只看终点成功，而是看中间如何形成通信与协调。」

---

# 7. Key Contributions（不要写空泛“总结”）

### 1. Dynamic Communication Modeling

```
Static / binary graph  →  Dynamic quality-weighted sparse graph
```

### 2. Long-horizon Stability

```
发现：图引导可抬短期 reward，却可能诱发长期 success 退化
提出：Residual decoupled policy（β 退火）
证据：Ablation 中 Residual 主导
```

### 3. Extensive Evaluation（证据链）

- Baseline comparison（16 UAV，5-seed）
- Ablation（机制）
- Scalability（4/8/16/32）
- Communication efficiency（Pareto）
- Zero-shot obstacle generalization
- Qualitative Demo（行为可视化）

---

# 8. 一句话收束

> **正式实验说明 DSGF 在困难多智能体设定下具有相对优势与可解释机制；Demo 只负责把动态图与残差协同“读给人看”。**

---

# Q&A 预备

| 问题 | 答法 |
|------|------|
| 提升大吗？显著吗？ | 16 UAV 上相对提升存在；p 未过 0.05，不夸张。关注机制 + 通信受限设定 |
| Demo 75% 和 Table 矛盾？ | Demo 是 4UAV、best episode、示意用；Table I 是 16UAV 多 seed 均值 |
| 为何用 20k？ | task-success 峰值可视化；论文正文仍报告完整训练 |
| full 通信 GAT 更高？ | 叙事是 **受限通信预算** 下的稳健性 |
| Ablation 9% 和 Table I 4%？ | **4 UAV vs 16 UAV，禁止混比** |

---

## 素材速查（科研汇报优先级）

| 优先级 | 文件 | 用途 |
|--------|------|------|
| ⭐⭐⭐⭐⭐ | `paper/figures/fig_pptx_dsgf_pipeline.png` | Method |
| ⭐⭐⭐⭐⭐ | `paper/figures/fig_pptx_motivation_degradation.png` | Motivation |
| ⭐⭐⭐⭐ | `demo/figures/comparison_trajectories.png` | 定性对比 |
| ⭐⭐⭐⭐ | `paper/figures/fig_pptx_checkpoint_selection.png` | Demo 严谨性 |
| ⭐⭐⭐⭐ | `paper/figures/fig1/4/6_*.png` | 正式结果 |
| ⭐⭐⭐ | `demo/figures/dsgf_behavior_stages.png` | 过程拆解 |
| 视频 | `demo/videos/comparison_mappo_gat_dsgf.mp4` | 放在结果之后 |

**完整实验冻结：** `paper/experiment_freeze.json`  
**本稿定位：** 科研汇报叙事版（v2）。
