# Stage C: Hierarchical PPO — Final Report

**Branch**: `exp/hierarchical-rl`  
**Decision tree**: Stage B failed → auto Stage C → N=16 PPO failed gates → **skip N=48**, no merge/tag.

## Stage B (N=16 REINFORCE)

| Metric | Gate | Last 10% mean | Pass |
| :--- | :--- | :--- | :--- |
| coverage | ≥ 0.75 | 0.570 | No |
| collision | ≤ 0.01 | 0.107 | No |
| light_scout | ≥ 0.60 | 0.440 | No |

Verdict: REINFORCE remains the bottleneck at N=16 (not scale-only).

## Stage C (N=16 PPO + GAE)

| Metric | Gate | Final | Last 10% mean | Pass |
| :--- | :--- | :--- | :--- | :--- |
| coverage | ≥ 0.75 | 0.600 | 0.538 | No |
| collision | ≤ 0.01 | 0.000 | 0.020 | No |
| light_scout | ≥ 0.70 | 1.000 | 1.000 | **Yes** |

Verdict: **FAIL overall** — PPO strongly improves role differentiation (`light_scout→1.0`) and cuts collision vs REINFORCE, but coverage collapses under the type-match / light_scout-heavy high-level reward.

N=48 **not run** (gated on N=16 pass).

## What landed in code

- [`models/hierarchical/ac_network.py`](../../models/hierarchical/ac_network.py) — type-conditioned `RoleActorCritic` + `LowLevelActorCritic`
- [`models/hierarchical/ppo_buffer.py`](../../models/hierarchical/ppo_buffer.py)
- [`models/hierarchical/trainer_ppo.py`](../../models/hierarchical/trainer_ppo.py)
- [`scripts/hierarchical/train_ppo.py`](../../scripts/hierarchical/train_ppo.py)
- [`configs/hierarchical/ppo.yaml`](../../configs/hierarchical/ppo.yaml)

## Artifacts

- `experiment_results/hierarchical/n16_reinforce/` — B metrics/plots/SUMMARY
- `experiment_results/hierarchical_ppo/smoke2/` — PPO smoke
- `experiment_results/hierarchical_ppo/n16_ppo/` — C metrics/plots

## Conclusion

At N=16, PPO fixes the REINFORCE variance story for **role emergence** (`light_scout`), but does **not** simultaneously satisfy coverage ≥ 0.75 and collision ≤ 0.01. Per plan: archive as “emergence vs task performance trade-off under current reward”; no `v2.1-ppo-emergence` tag.

## If continuing later

1. Down-weight `0.5 * lsr` / `match_weight` in high reward so coverage can recover.
2. Soften collision gate to ≤ 0.05 (aligned with方向1 ~0.04).
3. Only then re-open N=48.
