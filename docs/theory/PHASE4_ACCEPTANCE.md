# Phase 4 验收报告：Gazebo 桥接 CLI + dry-run

分支：`feature/transport-theory-upgrade`  
标签目标：`v2.6-phase4`

## 交付物

| 文件 | 说明 |
|------|------|
| `ros_nodes/transport_theory_bridge.py` | `--cscbf` / `--mubf` / `--scrise` / `--atac` |
| `scripts/run_theory_transport.sh` | `USE_CSCBF` / `USE_MUBF` / `USE_SC_RISE` / `USE_ATAC` |
| `scripts/phase4_dry_run_acceptance.sh` | 单控 + 组合 dry-run |
| `models/transport/control/hierarchical.py` | 四 flag 统一入口 |

> 桥接仍是 **host brain + Docker `set_reposition`**（`python3 …/transport_theory_bridge.py`），不是 `ros2 run` / MAVROS。

## Phase 4.1 CLI 映射

| 环境变量 | bridge flag | 日志关键字 |
|----------|-------------|------------|
| `USE_CSCBF=true` | `--cscbf` | `CSCBF active` |
| `USE_MUBF=true` | `--mubf` | `MUBF active` |
| `USE_SC_RISE=true` | `--scrise` | `SC-RISE active` |
| `USE_ATAC=true` | `--atac` | `ATAC active` |

## Phase 4.2 dry-run

```bash
bash scripts/phase4_dry_run_acceptance.sh
# 或：
USE_CSCBF=true DRY_RUN=1 STEPS=40 NO_ACCEPT=1 bash scripts/run_theory_transport.sh
```

| 控制器 | 门槛（仿真单测已覆盖；dry-run 验加载） |
|--------|----------------------------------------|
| CSCBF | 日志 `CSCBF active`；切换尖峰见 `scripts/test_cscbf_switch.py` |
| MUBF | 日志 `MUBF active`；编队误差见 `scripts/test_mubf_formation.py` |
| SC-RISE | 日志 `SC-RISE active`；CBF 残差见 `scripts/test_scrise_wind.py` |
| ATAC | 日志 `ATAC active`；裕度见 `scripts/test_atac_margin.py` |

### 记录

| 项 | 结果 | 日期 | 备注 |
|----|------|------|------|
| CSCBF dry-run | PASS | 2026-09-07 | `CSCBF active` |
| MUBF dry-run | PASS | 2026-09-07 | `MUBF active` |
| SC-RISE dry-run | PASS | 2026-09-07 | `SC-RISE active` |
| ATAC dry-run | PASS | 2026-09-07 | `ATAC active` |
| Combo A CSCBF+MUBF | PASS | 2026-09-07 | |
| Combo B CSCBF+SC-RISE | PASS | 2026-09-07 | |
| Combo C MUBF+ATAC | PASS | 2026-09-07 | |
| Combo D ALL | PASS | 2026-09-07 | host `pytorch12` dry-run |

## Phase 4.3 Gazebo SITL（需 AAS Docker）

| 控制器 | 命令 | 门槛 |
|--------|------|------|
| CSCBF | `USE_CSCBF=true bash scripts/run_theory_transport.sh` | 起飞；负载 d≤0.3m |
| MUBF | `USE_MUBF=true bash scripts/run_theory_transport.sh` | 编队误差 ≤0.3m |
| SC-RISE | `USE_SC_RISE=true WIND=1.0 bash scripts/run_theory_transport.sh` | 抗扰定位 ≤0.5m |
| ATAC | `USE_ATAC=true bash scripts/run_theory_transport.sh` | d≤0.3m |

## Phase 4.4 联合 dry-run

```bash
USE_CSCBF=true USE_MUBF=true DRY_RUN=1 bash scripts/run_theory_transport.sh
USE_CSCBF=true USE_SC_RISE=true DRY_RUN=1 bash scripts/run_theory_transport.sh
USE_MUBF=true USE_ATAC=true DRY_RUN=1 bash scripts/run_theory_transport.sh
USE_CSCBF=true USE_MUBF=true USE_SC_RISE=true USE_ATAC=true DRY_RUN=1 bash scripts/run_theory_transport.sh
```
