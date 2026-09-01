# UAM / AC-DSGF 项目索引（给 Agent 与自己）

> **交付包 + Obsidian 知识库：** `UAM-Vault/`（`powershell -File scripts/build_uam_vault.ps1` 重建）

> **2026-07-17 冻结：** 主线 = **AC-DSGF v1 投稿**；AC-DSGF++ = Supplement 负结果 / 未来工作  
> 详见 `paper/docs/freeze/AC_DSGF_FINAL_FREEZE.md`

## 一句话
- **v1（主论文）**：可学习通信拓扑 + 预算优化 + residual guidance（Success≈DSGF，Comm↓~90×）
- **++（补充材料）**：探索“消息是否值得发” — 实验未支撑主贡献，记录为 supervision 难点

## 快速入口

| 需求 | 去哪 |
|------|------|
| **最终冻结决议** | `paper/docs/freeze/AC_DSGF_FINAL_FREEZE.md` |
| **6–12 月路线图** | `paper/docs/planning/POST_FREEZE_ROADMAP.md` |
| **模拟审稿 R1–R3** | `paper/docs/freeze/REVIEWER_MOCK_AND_CHECKLIST.md` |
| **投稿三件套** | `paper/ac_dsgf/COVER_LETTER.md` · `HIGHLIGHTS.md` · `CONTRIBUTION_STATEMENT.md` |
| 投稿清单 Week1–3 | `paper/docs/freeze/PAPER_SUBMISSION_CHECKLIST.md` |
| ++ Supplement S5 | `paper/docs/freeze/SUPPLEMENT_S5_CAUSAL_UTILITY.md` |
| v1 主张锁 | `paper/docs/freeze/AC_DSGF_FREEZE.md` |
| v1 资产快照 | `paper/docs/freeze/AC_DSGF_V1_SNAPSHOT.md` |
| v1 论文 TeX | `paper/ac_dsgf/` |
| v1 主表 | `paper/tables/table1_final.csv` |
| Demo | `demo/videos/ac_dsgf_dynamic_comm.mp4` |

## 代码地图

```
models/
  ac_dsgf.py                 # v1 冻结 — 勿改
  ac_dsgf_pp.py              # ++ 仅 supplement；勿并入主结论
  communication/
    controller.py            # v1 gate — 勿改
    budget_layer.py          # v1 hard Top-K
    causal_utility*.py       # ++ only
configs/
  ac_dsgf/                   # v1
  ac_dsgf_pp/                # ++ exploratory (frozen as negative result)
```

## 训练入口（对照 / 复现 only）

```bash
# v1 — 冻结对照，勿调参
python train.py --exp configs/ac_dsgf/ac_dsgf_smoke_v0.yaml
```

**禁止：** 为投稿再开 ++ 超参搜索、16UAV 追 Success、改 v1 λ/架构。

## 结果目录
| 前缀 | 含义 |
|------|------|
| `results/ac_dsgf/` | v1（主证据） |
| `results/ac_dsgf_v1_freeze/` | v1 指针 |
| `results/ac_dsgf_pp/` | ++ 探索记录（Supplement） |

## 文档
| 子目录 | 内容 |
|--------|------|
| `paper/docs/freeze/` | **冻结与投稿清单** |
| `paper/supplement/` | ++ S5 大纲 |
| `paper/docs/planning/` | 历史计划（++ roadmap 已归档） |
| `paper/docs/advisor/` | 导师汇报 |
