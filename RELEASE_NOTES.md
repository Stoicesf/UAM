# Release v2.0-swarm-complete

**发布日期：** 2026-09-03  
**分支：** `feature/dice-phase2`  
**状态：** ✅ 全量验收通过，工程冻结归档

## 系统能力速览

| 模块 | 能力 | 状态 |
|------|------|------|
| **ICPS** | 通感算一体化（信道/算力/资源分配） | ✅ |
| **DICE** | 去中心化涌现智能（角色/任务/自愈） | ✅ |
| **SEM** | 语义通信（L1/L2/L3 自适应门控） | ✅ |
| **Adversarial** | 对抗性逃逸者（RL 博弈，猎手回升至 1.0） | ✅ |
| **Explainability** | 点击 UAV 回溯决策日志（WP3） | ✅ |
| **Lifelong** | 5000 步长期演化 + 20% 战损自愈 | ✅ |
| **Heterogeneous** | 异构机型（heavy/standard/light）支持，角色-机型匹配通过 **先验奖励引导（match_bonus）** 实现。专项训练实验表明，在纯任务奖励下轻型 UAV 未稳定收敛至侦察角色（终局 light_scout=0.50，处于随机水平），因此保留先验引导作为实际部署的合理方案。 | ✅ |
| **CBF Shield** | 控制屏障函数投影盾（碰撞率 0.0） | ✅ |
| **Gazebo Bridge** | ROS2 闭环桥接（真机仿真就绪） | ✅ |
| **Scaling** | 抽象仿真 N≤128 无 OOM | ✅ |

## 快速复现（冒烟验证）

```bash
conda activate pytorch12
pip install -e .

# 异构可视化
python demos/interactive_demo.py --scene search --hetero --n_agents 24

# CBF 零碰撞测试
python demos/interactive_demo.py --scene adversarial --shield cbf --n_agents 16

# Gazebo 闭环（需 AAS/WSL；默认 4 机）
bash scripts/run_aas_bridge.sh
```

> 异构可视化由环境 `reset()` 时的机型分配与角色先验偏置驱动，不依赖 `role_policy.pt`（该权重为实验产物，见 `experiment_results/hetero_training/archive/`）。

## 异构训练（先验引导，非纯涌现）

```bash
python scripts/train_role_emergence.py --hetero --steps 50000 --save_dir experiment_results/hetero_training/
# 去匹配先验的对照（预期不会出现稳定“轻型=侦察”涌现）
python demos/interactive_demo.py --scene search --hetero --n_agents 24 --no_match_bias
```

## 完整实验重现

```bash
# 全量语义对比
python -m experiments.semantic.semantic_experiments

# 迁移矩阵
python scripts/transfer_matrix.py --episodes 20

# 长期演化
python demos/lifelong_demo.py --steps 5000 --save_plot
```

## 实验备注

- **异构涌现训练**：`scripts/train_role_emergence.py` 在 `match_w=0` 后收敛至随机水平，表明当前任务奖励结构不足以驱动纯涌现角色分化。异构能力匹配在实际使用中依赖 `capability_match_bonus` 先验。

## 已归档研究记录：角色涌现与覆盖-碰撞帕累托平衡（2026-09）

本实验链以 `v2.0-swarm-complete` 为基线，旨在探索异构蜂群中“轻型 UAV 自发成为侦察者”的涌现机制，以及覆盖效率与碰撞安全之间的平衡边界。

### 实验分支

- `exp/emergence-v2`：方向1（安全探索奖励）
- `exp/hierarchical-rl`：方向2（分层RL + REINFORCE）
- `exp/hierarchical-ppo`：分层RL + PPO/GAE + 覆盖-碰撞帕累托调参（归档标签 `archived/emergence-ppo`）

### 关键发现

1. **角色分化可解**：PPO + 高层决策架构可实现 `light_scout=1.0`（轻型 UAV 全部成为侦察者）。
2. **覆盖-碰撞平衡存在结构性帕累托边界**：在 N=16 的 VMAS 环境中，无法同时满足 `coverage ≥ 0.70`、`collision ≤ 0.01`、`light_scout ≥ 0.70` 三项硬门槛。
3. **最优折衷点**：`coverage ≈ 0.59`、`collision ≈ 0.027`、`light_scout = 1.0`（奖励公式 `1.5*coverage - 0.5*collision`）。

### 归档结论

- 上述实验分支**不合并**至主干。
- `v2.0-swarm-complete` 保持为当前唯一稳定发布版本。
- 详细报告见 `experiment_results/ppo_tuning/FINAL_REPORT.md`（位于 `exp/hierarchical-ppo` 分支 / 标签 `archived/emergence-ppo`）。

### 后续建议

若未来继续探索该方向，建议：

- 换用更高保真度环境（Gazebo）重新验证帕累托前沿是否一致；
- 或在任务定义中放宽碰撞约束（如接受 `coll ≤ 0.03`），以换取更高的覆盖和分化。
