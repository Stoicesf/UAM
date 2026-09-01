# 代码

本目录为**交付快照**。日常训练请在仓库根 `F:\UAM` 运行（`python train.py --exp ...`），勿依赖本快照作为唯一工作区。

## 代码地图（对齐 INDEX）

```
src/
  models/ac_dsgf.py          # v1 冻结 — 勿改
  models/ac_dsgf_pp.py       # ++ 仅 supplement
  models/communication/      # gate / budget / causal_utility*
  guidance/                  # DSGF 图与引导
  algorithms/                # MAPPO baseline + guided
  env/ reward/ trainer/ tro/ secdo/ utils/ visualization/ experiments/
configs/
  ac_dsgf/                   # v1
  ac_dsgf_pp/                # ++ exploratory
  experiments/               # stage yaml
scripts/                     # 分析 / 画图 / 流水线
train.py evaluate.py test_env.py requirements.txt
```

## 复现入口（在仓库根）

```bash
conda activate dpg_hrl   # 或项目实际环境
cd F:\UAM
python test_env.py
python train.py --exp configs/ac_dsgf/ac_dsgf_smoke_v0.yaml
```

回到 [[00-首页]]。
