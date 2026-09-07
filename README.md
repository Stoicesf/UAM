# Multi-UAV Cooperative Transport

**从理论控制到物理迁移的完整框架**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-v2.6--phase4-brightgreen.svg)](https://github.com/Stoicesf/UAM)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 一句话定位

多无人机协同吊运系统的**完整理论与工程框架**，集成了 **混合系统建模、通用障碍函数（UBF）、RISE 抗扰控制、微分平坦规划、CBF/APF 双模安全盾、自适应缆绳构型（ATAC）** ，在 VMAS 抽象仿真和 Gazebo 物理引擎中均完成验证。

状态：`v2.6-phase4` · 四分支并行 · dry-run 验收 8/8（见 [`docs/theory/PHASE4_ACCEPTANCE.md`](docs/theory/PHASE4_ACCEPTANCE.md)）

---

## 核心特性

| 模块 | 功能 | 验证状态 |
| :--- | :--- | :--- |
| **混合系统动力学** | 缆绳 slack/taut 状态切换 + 非弹性碰撞重置 | ✅ VMAS + Gazebo |
| **UBF / M-UBF 控制** | 保证位置/编队跟踪误差的显式性能边界 | ✅ 编队误差 < 0.04m |
| **RISE / SC-RISE 抗扰** | 积分符号鲁棒控制 + CBF 安全融合 | ✅ 0.5N 风扰下误差 < 0.30m |
| **微分平坦规划** | 最小 snap 轨迹生成（scipy 可选） | ✅ 加速度连续 < 0.85 |
| **CBF + 指数型 APF** | 形式化安全 + 启发式避障双模 | ✅ 最小间距 1.12m |
| **ATAC 自适应构型** | 动态调整缆绳挂点/长度优化容量裕度 | ✅ 裕度 -1.17 → -0.23 |
| **Gazebo 物理迁移** | AAS Docker 桥接 + `set_reposition` 注入 | ✅ 负载精度 0.06m |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          协同运输控制架构（四层）                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Layer 4: 轨迹规划层（微分平坦 + minimum-snap）                             │
│  ├── 平坦输出: (x_L, R_L, Λ, ψ_i)                                         │
│  └── 11 阶多项式轨迹，加速度连续                                            │
│                                                                             │
│  Layer 3: 负载控制层（UBF / M-UBF + RISE / SC-RISE）                       │
│  ├── UBF 位置环: 保证跟踪误差 ≤ 预定义边界                                  │
│  ├── M-UBF 编队环: 同时约束绝对误差 + 相对误差                              │
│  └── SC-RISE 抗扰: 融合 CBF 的鲁棒积分控制                                  │
│                                                                             │
│  Layer 2: 编队与安全层（引力势场 + CSCBF/APF）                              │
│  ├── 引力势场编队: 加速收敛（η 增益可调）                                   │
│  ├── CSCBF: 混合系统形式化安全盾（slack/taut 感知）                         │
│  └── 指数型 APF: 软避障引导 + 局部极小逃离                                  │
│                                                                             │
│  Layer 1: 物理层（混合系统动力学 + ATAC）                                   │
│  ├── HybridPayloadDynamics: slack/taut 状态机 + 非弹性碰撞重置              │
│  └── ATAC Optimizer: 在线调整缆绳构型最大化 wrench 裕度                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  仿真/部署管道                                                               │
│                                                                             │
│   VMAS (抽象 2D)  →  TransportHierarchicalController  →  Gazebo (AAS)     │
│       (验证逻辑)            (统一控制器接口)              (物理验证)         │
│                                                                             │
│   支持 DRY_RUN=1 模式：无需 AAS Docker，直接验证控制逻辑                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 环境要求

- Python 3.10+ / PyTorch 2.0+
- （可选）AAS + Docker + ROS2 Jazzy（用于 Gazebo 物理仿真）

```bash
# 1. 克隆仓库
git clone https://github.com/Stoicesf/UAM.git
cd UAM

# 2. 安装依赖
conda create -n pytorch12 python=3.10
conda activate pytorch12
# 建议：先按 requirements.txt 顶部注释安装 CUDA 版 torch，再：
pip install -r requirements.txt
pip install -e .

# 或使用冻结环境：
# conda env create -f environment_frozen.yml && conda activate pytorch12 && pip install -e .

# 3. 运行 VMAS 演示（理论控制器）
python demos/interactive_demo.py --scene transport --ctrl theory --hybrid --steps 100

# 4. 启用所有理论控制器（CSCBF + MUBF + SC-RISE + ATAC）
python demos/interactive_demo.py --scene transport --ctrl theory --hybrid \
  --cscbf --mubf --scrise --atac --steps 100

# 5. Dry-run（无需 Gazebo 环境验证桥接）
USE_CSCBF=true USE_MUBF=true USE_SC_RISE=true USE_ATAC=true DRY_RUN=1 \
  bash scripts/run_theory_transport.sh

# 6. Gazebo 物理仿真（需 AAS Docker）
USE_CSCBF=true USE_MUBF=true USE_SC_RISE=true USE_ATAC=true \
  bash scripts/run_theory_transport.sh
```

---

## 研究分支（四理论方向）

本项目并行推进四个理论方向，均已实现并集成到 `feature/transport-theory-upgrade`：

| 分支 | 方向 | 核心理论 | 代码路径 | 验收数据 |
| :--- | :--- | :--- | :--- | :--- |
| `feature/theory-hybrid-formal` | 混合系统形式化 | CSCBF（混合状态感知 CBF） | `models/transport/safety/cscbf_shield.py` | 切换尖峰 7.4→1.1 m/s |
| `feature/theory-mubf` | 多智能体 UBF | M-UBF（绝对+相对误差联合约束） | `models/transport/control/mubf_controller.py` | 编队误差 < 0.04m |
| `feature/theory-scrise` | 安全-critical RISE | RISE + CBF 融合抗扰 | `models/transport/control/scrise_controller.py` | 风扰下 CBF 残差 ≥ 0 |
| `feature/theory-atac` | 自适应缆绳构型 | 容量裕度在线优化 | `models/transport/atac/atac_optimizer.py` | 裕度 -1.17→-0.23 |

---

## 性能基准（验收数据）

### VMAS 抽象仿真

| 指标 | 启发式基线 | 理论控制器（All） | 提升 |
| :--- | ---: | ---: | ---: |
| 负载到达距离（50 步） | 1.23m | **0.27m** | **78%** |
| 编队误差 | 0.15m | **0.04m** | **73%** |
| 0.5N 风扰下负载距离 | 5.86m (无 RISE) | **0.30m** (SC-RISE) | **95%** |
| 缆绳切换尖峰 | 7.4 m/s | **1.1 m/s** | **85%** |

### Gazebo 物理仿真

| 指标 | 理论控制器（All） | 门槛 |
| :--- | ---: | ---: |
| 负载到达距离 | **0.06m** | < 1.0m |
| 编队误差 | **0.08m** | < 0.3m |
| 碰撞率 | **0%** | 0% |

---

## 项目结构

```
UAM/
├── demos/                              # 交互演示
│   ├── interactive_demo.py
│   ├── scene_library.py
│   └── visualizer.py
│
├── environments/                       # 仿真环境
│   ├── dice_vmas_env.py               # VMAS 环境
│   ├── dynamics/
│   │   ├── hybrid_payload.py          # 混合系统动力学
│   │   └── transport_payload.py       # 弹簧模型（备选）
│   └── scenarios/
│       └── cooperative_transport.py    # 运输场景
│
├── models/transport/                  # 理论控制器模块
│   ├── control/
│   │   ├── hierarchical.py            # 统一控制器入口
│   │   ├── ubf_controller.py          # 单机 UBF
│   │   ├── mubf_controller.py         # M-UBF（分支二）
│   │   ├── rise_controller.py         # 基础 RISE
│   │   └── scrise_controller.py       # SC-RISE（分支三）
│   ├── formation/
│   │   └── attractive_potential.py    # 引力势场编队
│   ├── planning/
│   │   └── minimum_snap.py            # 微分平坦轨迹
│   ├── safety/
│   │   ├── cscbf_shield.py            # CSCBF（分支一）
│   │   ├── exponential_apf.py         # 指数型 APF
│   │   └── hybrid_shield.py           # CBF+APF 融合
│   └── atac/
│       └── atac_optimizer.py          # ATAC 优化器（分支四）
│
├── ros_nodes/                         # Gazebo 桥接
│   ├── transport_theory_bridge.py    # 理论控制器桥接
│   └── aas_transport_bridge.py       # 启发式桥接（备选）
│
├── scripts/
│   ├── run_theory_transport.sh       # Gazebo 启动脚本
│   ├── phase4_dry_run_acceptance.sh  # 批量验收
│   └── test_*.py
│
├── docs/theory/
│   ├── ROADMAP_FOUR_TRACKS.md
│   ├── PHASE4_ACCEPTANCE.md
│   ├── PHASE3_TO_PHASE6.md
│   ├── hybrid_formalization_survey.md
│   ├── mubf_survey.md
│   ├── scrise_survey.md
│   └── atac_survey.md
│
├── configs/
├── experiment_results/                # 实验数据（gitignored）
└── README.md                          # 本文件
```

同仓仍保留 ICPS+DICE+SEM 蜂群演示栈（`search` / `tracking` / `pursuit` 等 scene）；运输理论为主线，蜂群为既有产品面。

---

## 常用命令

```bash
# VMAS 演示
python demos/interactive_demo.py --scene transport --ctrl theory --hybrid --steps 100

# 启用全部理论控制器
python demos/interactive_demo.py --scene transport --ctrl theory --hybrid \
  --cscbf --mubf --scrise --atac --steps 100

# Dry-run（无 AAS）
USE_CSCBF=true USE_MUBF=true USE_SC_RISE=true USE_ATAC=true DRY_RUN=1 \
  bash scripts/run_theory_transport.sh

# Gazebo 物理仿真（需 AAS Docker）
USE_CSCBF=true USE_MUBF=true USE_SC_RISE=true USE_ATAC=true \
  bash scripts/run_theory_transport.sh

# 批量 dry-run 验收
bash scripts/phase4_dry_run_acceptance.sh

# 四分支切换
git checkout feature/theory-hybrid-formal  # 或 -mubf / -scrise / -atac
```

环境变量 ↔ bridge flag：`USE_CSCBF`→`--cscbf`，`USE_MUBF`→`--mubf`，`USE_SC_RISE`→`--scrise`，`USE_ATAC`→`--atac`。

---

## 路线图

| 阶段 | 状态 | 说明 |
| :--- | :--- | :--- |
| Phase 1 | ✅ | 文献调研 + 理论方向确定 |
| Phase 2 | ✅ | 四分支理论推导 + 代码骨架 |
| Phase 3 | ✅ | 四分支完整算法实现 + VMAS 验收 |
| Phase 4 | ✅ | Gazebo 桥接 CLI 集成 + dry-run 验收（8/8） |
| Phase 5 | 🔜 | Gazebo SITL 真实验收 + HITL 预研 |
| Phase 6 | 🔜 | 论文撰写与归档 |

详图见 [`docs/theory/ROADMAP_FOUR_TRACKS.md`](docs/theory/ROADMAP_FOUR_TRACKS.md)。

---

## 引用与致谢

本框架融合了以下理论方法：

- 混合系统建模：基于 Sreenath et al. (2013) 的微分平坦混合系统
- UBF 控制：基于 Tee et al. (2009) 的通用障碍函数
- RISE 控制：基于 Xian et al. (2004) 的鲁棒积分符号控制
- CBF：基于 Ames et al. (2019) 的控制屏障函数
- APF：基于 Khatib (1986) 的人工势场法

---

## License

MIT License — 见 [`LICENSE`](LICENSE)。

---

## 联系方式

问题与合作意向请提交 [GitHub Issues](https://github.com/Stoicesf/UAM/issues)。
