# AC-DSGF 完整任务规划

## Adaptive Communication-aware Dynamic Sparse Graph Fusion  
## 自适应通信感知动态稀疏图融合的多无人机蜂群协同框架

> **文档性质：** 下一代算法升级路线图（可执行任务规划）  
> **定位升级：**  
> - **DSGF v2（已冻结）** = 改进 MARL 图引导算法 · 支撑当前 IEEE 证据链  
> - **AC-DSGF（规划中）** = 面向真实蜂群**通信约束**的自主协同系统  
>  
> **总原则：**  
> 1. **不破坏** DSGF v2 冻结资产与现有 Table I–IV  
> 2. AC-DSGF = **在 DSGF 之上新增可学习通信决策层**，不是推倒重来  
> 3. Demo / 实验叙事围绕：`通信有限 → 智能选择通信 → 动态协同 → 任务完成`  
> 4. 目标不是“仿真 reward 再涨一点”，而是让导师一眼看出：**解决真实蜂群通信问题**

| 项目 | 内容 |
|------|------|
| 冻结基线 | DSGF v2 (`configs/dsgf/dsgf_v2_frozen.yaml`) |
| 新增层 | Communication Controller + Comm Cost + Uncertainty |
| 推荐投稿升级方向 | IEEE RA-L / TASE / TNNLS（通信约束协同） |
| 本文档对应代码骨架 | `models/communication/` · `models/ac_dsgf.py` · `models/uncertainty.py` |

---

# 一、最终算法定位

## 名称

| | |
|--|--|
| **英文** | **AC-DSGF** — Adaptive Communication-aware Dynamic Sparse Graph Fusion |
| **中文** | 自适应通信感知动态稀疏图融合的多无人机蜂群协同框架 |

## 一句话卖点

> 在通信预算与链路失效约束下，UAV 主动决定 **何时、与谁通信**，再经 DSGF 稀疏图融合与残差策略完成协同。

## 与 DSGF v2 的关系

```
DSGF v2 (frozen)
  = Dynamic Graph Ã=A·q
  + Sparse Attention
  + Temporal Memory
  + Residual Policy

AC-DSGF (next)
  = Communication Controller → g_ij
  + Ã' = A · q · g
  + L = L_ppo + λ_c · C_comm
  + Packet-loss / Budget evaluation & Demo
```

**保留全部已有实验作为 baseline / ablation 对照；不废弃 DSGF 工作。**

---

# 二、最终系统架构

```
                  Environment
                       |
          --------------------------------
          |                              |
     UAV local state                Communication state
          |                              |
          --------------------------------
                       |
              Communication Controller
                       |
              输出通信决策 g_ij ∈ [0,1]
                       |
          --------------------------------
          |                              |
 Dynamic Communication Graph        Communication Cost C=Σ g_ij
          |
       DSGF Encoder (v2 frozen modules)
          |
 --------------------------------
 |              |               |
Dynamic Graph Sparse Attention Temporal Memory
 |
 Residual Policy
 |
 Action → UAV motion
```

**叙事链（Demo / 论文统一）：**

`通信有限 → 智能选择通信(g_ij) → 动态协同(DSGF) → 任务完成`

---

# 三、代码改造总体结构

## 3.1 保留（禁止破坏冻结行为）

```
models/
├── dynamic_graph.py      # 可扩展 edge × g，默认 g=1 保持 DSGF
├── dsgf.py
├── guide_gnn.py
├── temporal_encoder.py
├── residual_policy.py
├── sparse_attention.py
```

## 3.2 新增（本规划落地目标）

```
models/
├── communication/
│   ├── __init__.py
│   ├── controller.py     # 通信决策网络 g_ij
│   ├── cost.py           # C = Σ g_ij
│   └── scheduler.py      # 预算 / 退火 / λ_c 调度
├── ac_dsgf.py            # 总模型入口
└── uncertainty.py        # 丢包 / 链路关闭扰动

configs/ac_dsgf/
├── ac_dsgf_v0.yaml       # smoke
├── ac_dsgf_16uav.yaml    # Table I'
├── budget_*.yaml         # 通信预算
└── packet_loss_*.yaml    # 通信故障

scripts/
├── smoke_ac_dsgf.py
├── run_ac_dsgf_main.py
├── eval_comm_budget.py
├── eval_packet_loss.py
└── demo_ac_dsgf.py

demo_ac_dsgf/
├── videos/
└── figures/

paper/
└── ac_dsgf_roadmap.md    # 本文档
```

---

# 四、模块设计规格

## 4.1 Communication Controller

**文件：** `models/communication/controller.py`

**学习目标：** UAV \(i\) 是否应与 UAV \(j\) 通信。

**输入（每条潜在边）：**

| 特征 | 说明 |
|------|------|
| \(h_i, h_j\) | 节点表征 |
| distance | \(\|p_i-p_j\|\) |
| signal_quality | 可复用 DSGF 的 \(q_{ij}\) / 信道先验 |
| task_priority | 可选：目标距离 / 拥挤度 |
| battery / resource | 可选（v1 可先常数或省略） |

**输出：** \(g_{ij}\in[0,1]\)（可用 Gumbel-Sigmoid / hard threshold at eval）

**伪代码：**

```python
class CommunicationController(nn.Module):
    def forward(self, h_i, h_j, edge_state):
        x = torch.cat([h_i, h_j, edge_state], dim=-1)
        return torch.sigmoid(self.mlp(x))  # g_ij
```

**示例可视化目标：**

```
UAV1–UAV2 : 0.92
UAV1–UAV3 : 0.08
UAV1–UAV4 : 0.65
```

## 4.2 Dynamic Graph 升级

**原 DSGF：** \(\tilde A_{ij} = A_{ij}\, q_{ij}\)  
**AC-DSGF：** \(\tilde A'_{ij} = A_{ij}\, q_{ij}\, g_{ij}\)

**实现约束：**

- `DynamicGraphModule` 增加可选 `comm_gate` 参数；`None` 时 ≡ 全 1 → **数值行为与 DSGF v2 一致**
- 硬半径 mask \(A_{ij}\) 仍生效（控制器不能凭空创造超视距边，除非单独做 relay 扩展）

## 4.3 Communication Cost

**文件：** `models/communication/cost.py`

\[
C_{\mathrm{comm}} = \sum_{i\neq j} g_{ij}
\quad\text{（或对称化后按无向边计数）}
\]

**对比叙事：** 10 UAV 全连接 ≈ 90 有向边（或 45 无向）；AC-DSGF 目标压到显著更少（Smoke 目标：**边数 ↓ ≥ 30%**，success 不塌）。

## 4.4 Scheduler

**文件：** `models/communication/scheduler.py`

职责：

- 通信预算 \(B\)（只允许 top-\(k\) 或 \(\sum g \le B\)）
- \(\lambda_c(t)\) warmup / 退火
- eval 时 hard gate（\(g\ge \tau\)）

## 4.5 Uncertainty / Failure

**文件：** `models/uncertainty.py`

- 独立丢包：每条边以概率 \(p\) 置零  
- 节点失效：关闭某 UAV 全部出边 / 入边  
- 仅 **eval / Demo**；训练默认关闭（或小概率 curriculum）

## 4.6 总模型入口

**文件：** `models/ac_dsgf.py`

```python
class ACDSGF(nn.Module):
    def forward(self, obs, positions=None, ...):
        h = self.node_embed(obs)
        q, A = ...                     # 几何质量图
        g = self.controller(h, ...)  # 学习通信
        A_tilde = A * q * g
        phi = self.dsgf_encoder(h, A_tilde, ...)  # 复用 DSGF 块
        # residual policy 在 guided MAPPO 侧
        return phi, g, A_tilde
```

## 4.7 训练目标

\[
\mathcal{L} = \mathcal{L}_{\mathrm{PPO}} + \lambda_c \, C_{\mathrm{comm}}
\]

- \(\lambda_c\) 过大 → 不通信、任务失败  
- 过小 → 退化为近全半径图（≈ DSGF）  
- Smoke 先网格 \(\lambda_c \in \{10^{-4},10^{-3},10^{-2}\}\)

**集成点：** `algorithms/guided/runner.py`（仅 `guidance.mode == ac_dsgf` 分支追加 comm loss；**不影响** `dsfg` 模式）

---

# 五、实验路线（三阶段 · 不重跑全部旧实验）

## Phase 0 — Smoke（约 2 天）⭐ 最先做

| 项 | 设定 |
|----|------|
| 环境 | 4 UAV · navigation |
| 帧数 | 20k |
| 对比 | **DSGF v2** vs **AC-DSGF** |
| 指标 | Success / Reward / Comm Cost / #Edges |

**通过标准：**

1. 代码可训练、可加载、可记 `communication.csv`  
2. 相对 DSGF，平均边数 **下降 ≥ 30%**  
3. Success 不明显崩盘（允许小幅波动，禁止“不通信躺平”）

**产出：** `results/ac_dsgf/smoke_*` · 边数柱状图

---

## Phase 1 — 核心论文实验（约 2–3 周）

### Exp-1 主结果（Table I′）

| 项 | 设定 |
|----|------|
| 环境 | **16 UAV** · 102k · seed=42（后补 3–5 seed） |
| 方法 | MAPPO · GAT · Transformer · **DSGF** · **AC-DSGF** |
| 指标 | Success / Collision / Reward / **Comm Cost** |

**期望叙事：** AC-DSGF ≈ 或略优 DSGF success，**Comm Cost 明显更低**。

> 现有 Table I（无 AC）完整保留为历史基线；新表可标 “with adaptive communication”。

### Exp-2 通信预算（**最重要 · 最大卖点**）

预算档位：`100% / 80% / 60% / 40% / 20%`（相对满半径可行边或相对 DSGF 平均边数定义，**需在实施时写死公式**）。

对比：MAPPO / GAT / DSGF / AC-DSGF  

**目标曲线形态：** 低预算下 AC-DSGF 掉点最少。

### Exp-3 通信故障 / 丢包

丢包率：`0 / 20% / 40% / 60% / 80%`  

曲线：Success vs packet loss —— AC-DSGF 更平。

### Exp-4 规模扩展

在 4/8/16/32 上补跑 **AC-DSGF 单曲线**（DSGF 数字直接复用已有 scalability）。

### Exp-5 可解释通信图

导出 \(g_{ij}(t)\) gif：展示 AI **主动改边**（t=0 与 t=50 拓扑不同）。

### Exp-6 Ablation（Table II′）

```
Full AC-DSGF
w/o Communication Controller   (g=1)
w/o Dynamic Quality q
w/o Temporal
w/o Residual
```

---

## Phase 2 — Demo（约 1–2 周）

### 标题

**Adaptive Communication-aware Swarm Coordination**

### 场景建议：通信受损搜索 / 导航

- 4 UAV · 可带 obstacle  
- 中途关闭 UAV2 通信（或突发高丢包）  

### 三栏对照

| | MAPPO | DSGF | AC-DSGF |
|--|-------|------|---------|
| 期望 | 协调崩溃 / 碰撞↑ | 尚能完成但边多 | **重选链路、降边、完成任务** |

### 输出目录

```
demo_ac_dsgf/
├── videos/
│   ├── mappo_failure.mp4
│   ├── dsgf_baseline.mp4
│   ├── ac_dsgf_success.mp4
│   └── comparison.mp4
└── figures/
    ├── communication_graph.gif
    ├── trajectory_compare.png
    └── communication_cost.png
```

**Demo 口径（继承 DSGF 汇报纪律）：**  
正式实验 = 有效性；Demo = 行为合理性。不拿单 episode success 替代 Table。

---

## Phase 3 — 论文与投稿（约 4–6 周）

### 论文图表骨架

| 图表 | 内容 |
|------|------|
| Figure 1 | AC-DSGF 系统框架（含 Communication Controller） |
| Figure 2 | **通信预算曲线**（核心卖点） |
| Figure 3 | 规模扩展 Success / Comm |
| Figure 4 | 动态通信图 \(g_{ij}(t)\) |
| Figure 5 | 丢包鲁棒性 |
| Table I′ | 主结果（含 Comm Cost） |
| Table II′ | Ablation |
| Table III | Complexity（沿用并注明 controller 额外 \(O(N^2 d_c)\) 或稀疏采样） |

### Method 叙事升级

旧：DSGF 改进图引导 MARL  
新：**真实通信预算下的自适应协同系统**

---

# 六、90 天执行日历

| 周次 | 任务 | 交付物 | 完成标准 |
|------|------|--------|----------|
| **W1** | Controller + cost + `Ã'=A·q·g` + mode=`ac_dsgf` | 可训练分支 | DSGF 模式回归测通过 |
| **W2** | Loss `+ λ_c C` + logger + Phase 0 smoke | smoke 报告 | 边数 ↓≥30% |
| **W3** | Exp-2 预算协议冻结 + 4/16 UAV 试跑 | budget 曲线草稿 | 低预算差距可见 |
| **W4** | Exp-1 16 UAV 主跑 + Exp-3 丢包 | Table I′ 初稿 | AC vs DSGF 可讲通 |
| **W5** | Exp-4/5 + Ablation | Table II′ / 图4 | 完整消融 |
| **W6** | Demo 三栏 + graph gif | `demo_ac_dsgf/` | 导师直觉：在解决通信问题 |
| **W7–8** | Method + Experiments 写作 | LaTeX v1 | 框架图 + 预算图就位 |
| **W9–10** | Related Work / 投稿包装 | IEEE 初稿 | 叙事对齐“蜂群通信” |
| **W11–12** | 多 seed 补强 + 投稿 | 投稿包 | RA-L / TASE / TNNLS |

---

# 七、优先级清单（落地顺序）

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | 通信控制器 + 图门控 + loss | 没有这个没有 AC |
| P0 | Smoke 4 UAV | 2 天验收 |
| P1 | **通信预算实验** | 论文最大卖点 |
| P1 | Comm Cost 进 Table | 与 DSGF 差异可视化 |
| P1 | 故障 / 丢包曲线 | 系统感 |
| P2 | 16 UAV 主表 + Ablation | 完整证据 |
| P2 | AC Demo 三栏 | 导师观感 |
| P3 | 多 seed / 32 UAV AC 曲线 | 加强统计 |
| ❌ | 推倒 DSGF / 重做全部旧表 | 禁止 |
| ❌ | 先上 World Model / 大 Transformer | 高风险低回报 |

---

# 八、风险与缓解

| 风险 | 缓解 |
|------|------|
| \(\lambda_c\) 过强 → 不通信 | Smoke 网格；budget 约束代替纯惩罚 |
| Controller 学崩 → 全连或全断 | entropy / sparse prior；eval hard-top-k |
| \(O(N^2)\) 控制器过贵 | 仅对 \(A_{ij}=1\) 边算 \(g\)；或 kNN 候选 |
| 与 DSGF 数值纠缠 | `mode=dsfg` 路径零改动；CI smoke 双跑 |
| Demo 夸大 | 继续拆分“实验 vs 行为”口径 |

---

# 九、验收里程碑（给导师的检查点）

### M1（约第 2 周末）— Smoke 通过

- [ ] AC-DSGF 可训练  
- [ ] 边数相对 DSGF ↓≥30%  
- [ ] 一张边数对比图  

### M2（约第 4–5 周末）— 卖点曲线

- [ ] 通信预算图：低预算下 AC-DSGF 更稳  
- [ ] 丢包曲线草稿  

### M3（约第 6 周末）— Demo

- [ ] 三栏视频：MAPPO / DSGF / AC-DSGF  
- [ ] 动态 \(g_{ij}\) gif  

### M4（约第 8–12 周）— 投稿稿

- [ ] Method + 预算实验写入正文  
- [ ] DSGF 旧结果作为强 baseline 引用保留  

---

# 十、与当前 DSGF 资产映射（不浪费）

| 已有资产 | AC-DSGF 中角色 |
|----------|----------------|
| Table I DSGF 4.01% | 强 baseline |
| Table II Residual 消融 | 继续作为机制支撑；再加 w/o Controller |
| Table IV 半径扫描 | 预算实验的前身 / 对照 |
| Table V Complexity | 扩展一行 AC-DSGF |
| Demo MAPPO/GAT/DSGF | 左侧两栏可复用；右侧替换/新增 AC |
| `dsgf_v2_frozen.yaml` | **永不改**；AC 用新 config |

---

# 十一、实施状态板

| 模块 | 状态 | 备注 |
|------|------|------|
| 本路线图文档 | ✅ Done | `paper/ac_dsgf_roadmap.md` |
| `models/communication/*` 骨架 | ✅ Scaffold | 接口就位 |
| `models/ac_dsgf.py` | ✅ Wired | factory `mode=ac_dsgf` |
| guided runner `λ_c·C` | ✅ Done | 仅 `ac_dsgf` 分支 |
| `configs/ac_dsgf/ac_dsgf_smoke_v0.yaml` | ✅ Done | |
| Phase 0 Smoke 训练 | ✅ Done | success≈32.5%；soft edges≈0.27 vs DSGF≈6（↓≫30%）；需观察是否过稀疏 |
| Phase 1 核心实验 | ⏳ Pending | W3–W5 |
| Phase 2 Demo | ⏳ Pending | W6 |
| Phase 3 写作投稿 | ⏳ Pending | W7–W12 |

---

# 十二、判断（给团队的定位句）

| | DSGF v2（现在） | AC-DSGF（下一代） |
|--|-----------------|-------------------|
| 类别 | 改进多智能体强化学习算法 | **面向真实蜂群通信约束的自主协同系统** |
| Demo 观感 | 飞得更协调 | **在通信变差时仍会“省着用、换着用”并完成任务** |
| 风险 | 已冻结 | 低：只加通信决策层 |
| 论文价值跃迁 | IEEE 证据链已具备 | **通信预算 / 故障曲线**成为主卖点 |

---

**下一步立刻可执行的第一刀（不改冻结 DSGF）：**

1. 实现 `CommunicationController` + `Ã'=A·q·g`（`g=1` 时数值对齐 DSGF）  
2. `L += λ_c C` 仅挂在 `ac_dsgf`  
3. 跑 Phase 0 Smoke（4 UAV · 20k）  

完成 Smoke 后再进入预算实验——**那才是 AC-DSGF 的论文命门。**
