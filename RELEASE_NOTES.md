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
| **Heterogeneous** | 异构机型（heavy/standard/light）能力匹配 | ✅ |
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

# Gazebo 闭环（需 AAS 环境；默认 4 机）
bash scripts/run_gazebo_bridge.sh 4
```

## 异构涌现训练

```bash
python scripts/train_role_emergence.py --hetero --steps 50000 --save_dir experiment_results/hetero_training/
# 去硬偏置验收
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
