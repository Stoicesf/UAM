# Phase 1：多智能体协同 UBF（M-UBF）调研

分支：`feature/theory-mubf`  
对照代码：[`models/transport/control/ubf_controller.py`](../../models/transport/control/ubf_controller.py)、[`models/transport/formation/attractive_potential.py`](../../models/transport/formation/attractive_potential.py)

## 1. 背景

当前 UBF 仅对**负载绝对位置误差**做障碍变换与稳定化，输出期望力 \(F_d\)。多机吊运还需要相对约束：机间距离、相对编队误差、载荷偏角/缆绳夹角等。M-UBF 目标是把绝对跟踪与相对状态编进统一障碍函数族，并经 QP 与输入约束兼容。

## 2. 相关工作要点（调研提纲）

- **UBF / 障碍函数控制**：误差变换、性能边界 \(\delta\)、自适应扰动项。
- **多机相对约束**：连接保持、避碰、编队误差作为高阶约束。
- **CBF-QP / CLF-CBF-QP**：名义控制 + 线性约束二次规划；求解器选型（OSQP / `scipy`）。
- **吊运特有约束**：缆绳不可压、张力分配可行性（与方向四 capacity margin 接口）。

## 3. 相对本仓库现状的差距

| 现有实现 | M-UBF 缺口 |
|----------|------------|
| `UBFLoadController`：标量 \(d_{eL}\)、屏障球 \(\delta_{dL}\) | 仅绝对跟踪；无相对状态 \(\|p_i-p_j\|\) |
| 输出 \(F_d\in\mathbb{R}^2\)，再质量加权分张力 | 无 QP；张力/推力限幅在下游 clamp |
| 编队靠 `AttractivePotentialFormation` + `_ideal_ring` | 软吸引，非硬安全证书 |
| `HybridSafetyShield`：位置步投影 + APF | 与 UBF 环分离，非同一 Lyapunov/障碍框架 |
| 2D、`g=0` | 相对约束维数简单，但与 3D 文献不对齐 |

## 4. Phase 2–3 理论 / 工程问题清单

1. 定义多智能体误差：\(e_L\)（载荷）、\(e_{ij}\)（机间）、可选 \(e_{\mathrm{form}}\)（相对理想环）。
2. 构造 M-UBF 变换及复合 Lyapunov 候选；证明 ISS 或指数收敛（名义无扰动）。
3. 将 \(\dot h_k \ge -\alpha(h_k)\) 写成对控制输入仿射的 QP 约束。
4. 实现 `models/transport/control/mubf_qp.py`；默认 `use_mubf=False` 接入 hierarchical。
5. 依赖策略：优先纯 `torch`/`numpy`；若 QP 必需再引入 OSQP 并写清可选依赖。

## 5. 验收门槛

| 阶段 | 门槛 |
|------|------|
| Phase 1 | 约束类型清单 + 与现 UBF 差距清晰 |
| Phase 3 | `mubf_qp` 单元测试通过 |
| Phase 4 | 编队误差更小；机间距离 ≥ 安全阈值（如 0.5 m） |
| Phase 5 | Gazebo/AAS 编队误差 < 0.2 m，无碰撞 |

## 6. 论文章节指针

拟写入：“多智能体协同 UBF 控制”（M-UBF 定义 → 稳定性 → QP → 对比实验）。
