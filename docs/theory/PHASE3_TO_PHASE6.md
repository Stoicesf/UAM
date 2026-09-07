# Phase 3 → Phase 6 执行状态

更新于 Phase 4 dry-run 验收后。基线：`feature/transport-theory-upgrade`；对外文档见根目录 [`README.md`](../../README.md)（`v2.6-phase4`）。

## Phase 3 完成情况

| 分支 | tip（约） | 核心交付 | 验收脚本 |
|------|-----------|----------|----------|
| `feature/theory-hybrid-formal` | CSCBF apply + `use_cscbf` | `cscbf_shield.py` 完整投影；切换尖峰对比图 | `scripts/test_cscbf_switch.py` |
| `feature/theory-mubf` | `act()` + `use_mubf` | nominal + pairwise 投影；编队/间距对比 | `scripts/test_mubf_formation.py` |
| `feature/theory-scrise` | `act()` + `use_scrise` | 工作空间球 CBF + 全增益修正；风扰残差对比 | `scripts/test_scrise_wind.py` |
| `feature/theory-atac` | `use_atac` + 几何修正 | 偏置挂点使半径影响 η；裕度对比 | `scripts/test_atac_margin.py` |

图表写入 `experiment_results/{hybrid_formal,mubf,scrise,atac}/`（目录被 `.gitignore`，本地生成即可）。

**刻意简化**：未引入 OSQP；QP 用闭式投影代替。

## Phase 4（已完成）：系统集成 + dry-run

详见 [`PHASE4_ACCEPTANCE.md`](PHASE4_ACCEPTANCE.md)。

1. `transport_theory_bridge.py` / `run_theory_transport.sh` / demo CLI：`--cscbf` / `--mubf` / `--scrise` / `--atac`
2. dry-run 单控 + 组合 **8/8 PASS**（2026-09-07）
3. Gazebo SITL 真实验收留待 Phase 5（需 AAS Docker）

## Phase 5（待办）：Gazebo SITL + HITL 预研

每分支约 2–3 周：AAS SITL 门槛跑通；HITL 需另开 Pixhawk 通路（当前为 Docker `set_reposition`）。

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
python scripts/test_cscbf_switch.py
python scripts/test_mubf_formation.py
python scripts/test_scrise_wind.py
python scripts/test_atac_margin.py
bash scripts/phase4_dry_run_acceptance.sh
```
