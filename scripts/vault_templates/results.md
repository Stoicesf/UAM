# 实验结果

完整 run（含 checkpoint）仍在仓库 `../results/`（约 1.3GB）。本区只保留轻量证据：`summary.json`、`metrics.csv`、`meta.json`、`config.yaml`、`TAG.txt`、`plots/`。

## 结果前缀

| 前缀 | 含义 |
|------|------|
| `ac_dsgf/` | v1 主证据 |
| `ac_dsgf_v1_freeze/` | v1 指针 |
| `ac_dsgf_pp/` | ++ 探索（Supplement） |
| `baseline16/` / `baseline16_seeds/` | 5-seed 对照 |
| `guide/` / `graph/` / `dsgf/` / `sparse/` | Stage 流水线 |
| `scalability/` / `communication/` / `generalization/` | 论文实验轴 |
| `secdo/` / `secdo_v2/` | SECDO 线 |

## 本区内容

- [[results-README|results/README]] — 结果目录约定
- `tables/` — 论文表（自 `paper/tables`）
- `runs/` — 按前缀分类的轻量 run 摘要
- `figures/` — `paper/figures` + 有 plots 的 run 图

回到 [[00-首页]]。
