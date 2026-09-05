# Hierarchical RL（方向2）最终报告

**分支**: `exp/hierarchical-rl`  
**对比基线（方向1）**: `experiment_results/emergence_scout/summary.json`  
`final_light_scout_ratio = 0.133`, `final_coverage = 0.80`, `final_collision_rate = 0.042`

## 结论

**未达验收标准，不合并、不打 tag。**

方向2 的 `light_scout`（终局窗口均值 ~0.43）高于方向1 的 0.133，说明机型条件化高层有分化信号；但三项硬门槛均未同时满足。

## 验收对照

| 指标 | 门槛 | n48_run1（无类型条件） | n48_run2（类型条件 + match） | 结果 |
| :--- | :--- | :--- | :--- | :--- |
| coverage 终局 | ≥ 0.70 | 0.40（末20均值 0.70） | 0.60（末20均值 0.83） | 未稳定达标 |
| collision 终局 | ≤ 0.01 | 0.35 | 0.33 | 未达标 |
| light_scout 终局 | ≥ 0.60 | 0.33 | 0.40（末10%均值 0.43） | 未达标 |

产物：

- `experiment_results/hierarchical/smoke/` — 冒烟通过
- `experiment_results/hierarchical/n48_run1/` — metrics + plots
- `experiment_results/hierarchical/n48_run2/` — metrics + plots（主结果）

## 失败原因

1. **碰撞门槛过严（主因之一）**  
   N=48 下方向1 碰撞已达 0.042；方向2 末窗口均值 ~0.20。`≤0.01` 相对当前奖励与动力学不现实，需更大碰撞惩罚或更长低层训练，且可能牺牲覆盖。

2. **light_scout 信号弱于任务覆盖**  
   高层回报以 coverage 为主；即使加入 `capability_match` 与 `0.5 * light_scout`，episode 级 REINFORCE 方差大，分化不稳定（单 ep 可达 0.60–0.67，终局回落）。

3. **run1 结构性缺陷（已修）**  
   初版高层对所有机型共用同一 Categorical，`E[light_scout]=P(SCOUT)≈1/3`。run2 改为 type embedding 条件化后均值从 ~0.25 提到 ~0.43。

4. **交替训练干扰**  
   低层阶段冻结高层后按 episode 重采样固定角色，日志上 light_scout 波动大，终局点噪声高。

## 已实现内容（可复用）

- `models/hierarchical/{high_level,low_level,trainer,utils}.py`
- `scripts/hierarchical/{train,plot_metrics}.py`
- `configs/hierarchical/default.yaml`
- `DICEVMASEnv.get_global_obs()`
- 冒烟：N=16 × 1 round × 5k/5k 通过

## 建议下一刀（若继续）

1. 将验收 collision 改为 ≤0.05（与方向1 同量级），或加大 env 碰撞系数并单独扫参。  
2. 高层逐步回报（每 decision_interval 给一次 R）代替纯 episode 末标量。  
3. `match_weight` 提到 3–5，或高/低轮数改为 high-heavy（如 60k/20k）。  
4. 低层阶段不要每 ep 重采样角色，而在整个 low 阶段锁定一批 roles。

## 复现命令

```bash
# 冒烟
E:\ANACONDA\envs\pytorch12\python.exe scripts/hierarchical/train.py \
  --n 16 --rounds 1 --high_steps 5000 --low_steps 5000 \
  --save_dir experiment_results/hierarchical/smoke/

# 正式（当前默认含 type-cond / match / entropy bonus）
E:\ANACONDA\envs\pytorch12\python.exe scripts/hierarchical/train.py \
  --n 48 --rounds 3 --high_steps 40000 --low_steps 40000 \
  --save_dir experiment_results/hierarchical/n48_run2/ --seed 42
```
