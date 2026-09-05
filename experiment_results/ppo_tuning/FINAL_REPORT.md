# PPO 覆盖–碰撞帕累托调参报告

**分支**: `exp/hierarchical-ppo`  
**环境**: conda `pytorch12` + CUDA (RTX 4060)  
**前提**: 接受角色分化已被 PPO 解决；本轮只调 `coverage`/`collision` 奖励与 GAE/网络。

## 代码改动

- [`scripts/hierarchical/train_ppo.py`](../../scripts/hierarchical/train_ppo.py): `--reward_formula` / `--gamma` / `--gae_lambda` / `--device cuda` / `--hidden`
- [`models/hierarchical/utils.py`](../../models/hierarchical/utils.py): `eval_reward_formula`（AST 白名单，仅 `coverage`/`collision`）
- [`models/hierarchical/trainer_ppo.py`](../../models/hierarchical/trainer_ppo.py): GPU 张量路径；高层 R 用公式；关闭 lsr/match 塑形

> 说明：全局 coverage–collision 奖励作用在**高层决策回报**（及低层 0.1 倍 shaping），而非改写 `env.step` 内任务奖励；这是与方案意图等价、且不破坏环境动力学的最小落点。

## 验收门槛

| 级别 | coverage | collision | light_scout |
| :--- | :--- | :--- | :--- |
| 硬门槛 | ≥ 0.70 | ≤ 0.01 | ≥ 0.70 |
| 软门槛 | ≥ 0.65 | ≤ 0.01 | ≥ 0.70 |

**结果：硬门槛与软门槛均未达成 → 终止攻坚，不合并、不打 tag。**

## 帕累托前沿（末 10% 窗口均值）

| 实验 | cov | coll | lsr | 备注 |
| :--- | ---: | ---: | ---: | :--- |
| **E4** `1.5*cov - 0.5*coll` | **0.590** | 0.027 | **1.000** | **综合最优** |
| E5 `cov - 0.1*coll` | 0.553 | **0.012** | 1.000 | 碰撞最低 |
| E3 `cov - 1.0*coll` | 0.512 | 0.021 | 1.000 | |
| E1 `cov - 0.2*coll` | **0.700** | 0.031 | 0.000 | 覆盖达标但角色坍塌 |
| E2 `1.2*cov - 0.5*coll` | 0.478 | 0.037 | 0.000 | 最差 |
| GAE 扫（E4/E5 × γ/λ） | ≤0.53 | 0.015–0.042 | 0 或 1 | 未优于默认 E4 |
| **network_256_E4** | **0.709** | 0.101 | 0.000 | 覆盖破 0.70，但碰撞/分化双崩 |
| **network_256_E1** | 0.498 | **0.003** | 0.000 | 碰撞达标，覆盖与分化不足 |

完整表：[`pareto_summary.json`](pareto_summary.json)

## 结论

1. **分化与覆盖存在明显权衡**：E1 可到 cov≈0.70，但 `light_scout→0`；E4/E5 保住 lsr=1.0，覆盖卡在 ~0.55–0.59。
2. **碰撞 ≤0.01 在当前动力学+奖励下极难与高覆盖同达**：最优碰撞侧 E5 仍为 0.012；E4 为 0.027。
3. GAE（γ/λ）与 hidden=256 **未突破**帕累托前沿；瓶颈更可能在任务奖励/角色分配权衡，而非 PPO 方差。

## 决策（按方案决策树）

> 所有组合均 `cov < 0.65` 或 `coll > 0.02`（实际上 soft 要求 coll≤0.01 也全灭）  
> → **终止攻坚，归档帕累托前沿**；最优记录为 **E4**（cov≈0.59, coll≈0.027, lsr=1.0）。

不执行 merge / `v2.1-ppo-balanced`。
