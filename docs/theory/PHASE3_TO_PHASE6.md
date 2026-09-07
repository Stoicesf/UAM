# Phase 3 → Phase 6 执行状态

更新于 Phase 3 落地后。基线：`feature/transport-theory-upgrade` @ `7aff083`；各研究分支 tip 见下。

## Phase 3 完成情况（本轮已交付）

| 分支 | tip（约） | 核心交付 | 验收脚本 |
|------|-----------|----------|----------|
| `feature/theory-hybrid-formal` | CSCBF apply + `use_cscbf` | `cscbf_shield.py` 完整投影；切换尖峰对比图 | `scripts/test_cscbf_switch.py` |
| `feature/theory-mubf` | `act()` + `use_mubf` | nominal + pairwise 投影；编队/间距对比 | `scripts/test_mubf_formation.py` |
| `feature/theory-scrise` | `act()` + `use_scrise` | 工作空间球 CBF + 全增益修正；风扰残差对比 | `scripts/test_scrise_wind.py` |
| `feature/theory-atac` | `use_atac` + 几何修正 | 偏置挂点使半径影响 η；裕度对比 | `scripts/test_atac_margin.py` |

图表写入 `experiment_results/{hybrid_formal,mubf,scrise,atac}/`（目录被 `.gitignore`，本地生成即可）。

**刻意简化（相对原 4 周方案）**：未引入 OSQP；QP 用闭式投影代替；Gazebo/HITL **未做**（属 Phase 4–5）。

## Phase 4（待办）：系统集成与 Gazebo / AAS

每分支约 3 周：

1. `transport_theory_bridge.py` 增加 `--cscbf` / `--mubf` / `--scrise` / `--atac`（仅本分支 flag，避免冲突）
2. dry-run + 真机 SITL 各跑对照
3. 报告：`docs/theory/*_gazebo_report.md`

## Phase 5（待办）：HITL 预研

每分支约 2 周：AAS HITL + 日志归档。当前栈为 Docker `set_reposition`，HITL 需另开 Pixhawk 通路。

## Phase 6（待办）：论文交付

| 章节 | 来源分支 |
|------|----------|
| §3 混合系统与 CSCBF | hybrid-formal |
| §4 M-UBF | mubf |
| §5 SC-RISE | scrise |
| §6 ATAC / capacity margin | atac |

合并路径：`integration/YYYY-MM-DD` → `feature/transport-theory-upgrade` → 标签 `v3.0-theory-full`。

## 回归（集成分支必跑）

```bash
python scripts/test_hybrid_payload.py
python scripts/test_transport_phase2.py
python scripts/test_transport_phase34.py
python scripts/test_cscbf_switch.py      # on hybrid-formal
python scripts/test_mubf_formation.py    # on mubf
python scripts/test_scrise_wind.py       # on scrise
python scripts/test_atac_margin.py       # on atac
python ros_nodes/transport_theory_bridge.py --dry_run --steps 40
```
