# UAV-Swarm: DSGF-HRL

> **未知环境下基于动态蜂群引导场（DSGF）的层级强化学习无人机蜂群协同导航方法研究**

论文驱动开发（Paper-driven Development）— 整篇论文只回答一个问题：

**如何利用少量通信实现未知环境下的大规模无人机协同导航？**

---

## 算法对比（仅两个）

| 方法 | 角色 | 代码来源 |
|------|------|----------|
| **MAPPO** | Baseline | 复用（VMAS 官方 / TorchRL） |
| **DSGF-HRL** | 创新 | `guidance/` + `reward/guidance_reward.py` |

不要自己写：PPO、Transformer、GNN、LLM 全套框架。

---

## 训练数据流

```
VMAS → 状态 → 构建通信图 → DSGF → Φ → Actor(obs+Φ) → Action
     → Environment → Reward(Goal+Collision+Guide) → MAPPO Update
```

MAPPO 更新逻辑**不变**。真正新增的只有：

- `guidance/` — Graph + DSGF + Sparse Attention
- `reward/guidance_reward.py` — 方向对齐奖励

---

## 项目结构 ↔ 论文章节

| 目录 | 论文章节 | 说明 |
|------|----------|------|
| `guidance/` | 第三章 | **核心创新**，全部自己写 |
| `algorithms/` `models/` `reward/` | 第四章 | MAPPO 复用 + DSGF 包装 |
| `experiments/` | 第五章 | 5 个实验，每个一个脚本 |

---

## 环境

```bash
conda activate dpg_hrl
cd f:\UAM
python test_env.py          # 验证 VMAS + GPU
```

---

## 开发路线（8 周）

| 周 | 目标 | 关键文件 |
|----|------|----------|
| **Week 1** | VMAS + MAPPO baseline | `env/`, `algorithms/baseline/` |
| **Week 2** | DSGFEncoder | `guidance/dsfg_encoder.py` |
| **Week 3** | Guidance Reward | `reward/guidance_reward.py` |
| **Week 4** | 实验1 Baseline 对比 | `experiments/baseline.py` |
| **Week 5** | 实验5 消融 | `experiments/ablation.py` |
| **Week 6** | 实验4 可扩展性 | `experiments/scalability.py` |
| **Week 7** | 实验3 通信开销 | `experiments/communication.py` |
| **Week 8** | Isaac 验证（可选） | — |

### Day-by-Day（第一版 MVP）

| Day | 任务 |
|-----|------|
| Day 1 | 跑通 VMAS → `test_env.py` ✅ |
| Day 2 | 跑通 MAPPO baseline |
| Day 3 | 保存 baseline checkpoint |
| Day 4 | Actor 输入 obs + Φ（+4 维） |
| Day 5 | 实现 DSGFEncoder |
| Day 6 | 加入 Guidance Reward |
| Day 7 | 第一次完整训练 |

---

## 代码量预算

| 模块 | 自写？ | 预计行数 |
|------|--------|----------|
| VMAS 环境 | 复用 | 0 |
| MAPPO 框架 | 复用 | 100–300（适配） |
| Actor/Critic | 复用+改 | ~100 |
| DSGF 动态图 | ✅ | 200–300 |
| DSGF 稀疏注意力 | ✅ | 300–500 |
| Guidance 融合 | ✅ | 150–250 |
| Guidance 奖励 | ✅ | 100–150 |
| 实验脚本 | ✅ | 400–600 |
| **合计自写** | | **~1500–2500** |

---

## 五个实验（第五章）

| 实验 | 脚本 | 指标 |
|------|------|------|
| 1 Baseline | `experiments/baseline.py` | Success Rate |
| 2 碰撞 | 同上 | Collision Rate |
| 3 通信 | `experiments/communication.py` | Communication Cost |
| 4 可扩展性 | `experiments/scalability.py` | 4/8/16/32 agents |
| 5 消融 | `experiments/ablation.py` | 去掉 DSGF / Guide / Sparse Attn |

---

## Stage 0 — Baseline (DONE, frozen)

```bash
python train.py --exp configs/experiments/exp0_baseline.yaml
```

## Stage 1 — Guide MLP (CURRENT)

```bash
python train.py --exp configs/experiments/exp1_guide.yaml
python train.py --exp configs/experiments/exp1_guide.yaml --smoke
```

Compare: `exp0_baseline` vs `exp1_guide` -> first thesis figure.

## Experiment management

Each run auto-saves to `results/<run_name>/`:
- config.yaml, meta.json, metrics.csv, summary.json
- checkpoints/, tensorboard/

```bash
python scripts/plot_results.py
```

## Full stage roadmap

See `configs/experiments/README.md`



---

## DSGF 核心概念

DSGF **不是** Transformer 论文。它就是：

```
Swarm State → Sparse Attention → Compressed Guidance → Φ_i = [dx, dy, risk, priority]
```

Actor 输入：`obs (64维) + Φ (4维) = 68维`。MAPPO 不用改。

Reward：`R = R_goal + R_collision + cos(θ_action - θ_Φ)`
