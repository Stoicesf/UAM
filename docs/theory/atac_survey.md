# Phase 1：自适应缆绳构型（ATAC）调研

分支：`feature/theory-atac`  
对照代码：固定 `L0`、[`cooperative_transport._ideal_ring`](../../environments/scenarios/cooperative_transport.py)、桥接侧环阵 setpoint

## 1. 背景

当前系统把绳长 \(L_0\)、挂载几何（环阵半径）当作**常数**。ATAC（物理-控制协同）主张在任务中在线调整可重构参数（绳长、连接点、环半径）以提升 wrench 可行性裕度或降低峰值张力——形态即计算。

## 2. 相关工作要点（调研提纲）

- **绳系/吊运构型优化**：静定/超静定张力分配、可行 wrench 锥。
- **可变绳长 / 铰接点**：主动绞车、被动滑移；仿真中常先做参数化而非真插件。
- **在线优化**：贝叶斯优化、MPC、RL；与安全约束联立时需低维参数（避免实时维数爆炸）。
- **形态计算**：物理构型嵌入控制性能指标。

## 3. 相对本仓库现状的差距

| 现有实现 | ATAC 缺口 |
|----------|-----------|
| `PayloadDynamics` / `HybridPayloadDynamics` 固定 `L0` | 无时变绳长状态或输入 |
| `_ideal_ring`：质量偏置环 + lead | 几何由启发式生成，非优化 |
| 张力仅作观测/奖励惩罚 | 无 “capacity margin / wrench 可行性” 指标 |
| AAS 桥：虚拟绳 + `set_reposition` | **无 Gazebo 缆绳插件**；变绳长只能主机改 `L0` 再重算环 |
| 规划 `MinimumSnapTrajectory` 只规划载荷平面轨迹 | 不联立构型参数 |

### 可实现性分层（本仓库约束下）

1. **L1（优先）**：VMAS/主机侧时变 `L0(t)` 或环半径增益；指标 = 张力峰值、定位误差。  
2. **L2**：离散切换挂载角（重算 `formation.delta`）。  
3. **L3**：Gazebo 真绳插件 / 绞车——超出当前 AAS 桥架构，后置。

## 4. Phase 2–3 问题清单

1. 定义 capacity margin \(\mu\)（相对可行张力/加速度集的距离）。
2. 建立 \(\mu\) 与 \((L_0,\{\rho_i\})\) 的可计算近似（2D 先解析/数值）。
3. 设计慢时标优化（任务段内更新构型）+ 快时标跟踪（现有 hierarchical）。
4. 实现 `models/transport/atac/`（优化器）；`use_atac` flag 默认关。
5. AAS：仅 L1/L2 间接模拟；文档标明非物理绞车。

## 5. 验收门槛

| 阶段 | 门槛 |
|------|------|
| Phase 1 | 可调参数范围与 L1–L3 可行路径明确 |
| Phase 4 | 相对固定构型：定位精度 ↑≥10% 或峰值张力 ↓≥20% |
| Phase 5 | AAS/Gazebo 演示动态调参可行性（允许主机侧模拟） |

## 6. 论文章节指针

拟写入：“物理-控制协同优化”（capacity margin → ATAC 策略 → 对比实验）。
