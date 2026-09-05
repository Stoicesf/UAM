# Stage B: N=16 REINFORCE

**Command**: `--n 16 --rounds 3 --high_steps 30000 --low_steps 30000 --seed 42`  
**Branch**: `exp/hierarchical-rl`

## Metrics (primary = last 10% window mean)

| Metric | Gate | Final point | Last 20 mean | Last 10% mean | Pass |
| :--- | :--- | :--- | :--- | :--- | :--- |
| coverage | ≥ 0.75 | 0.800 | 0.560 | 0.570 | No |
| collision | ≤ 0.01 | 0.250 | 0.094 | 0.107 | No |
| light_scout | ≥ 0.60 | 0.500 | 0.367 | 0.440 | No |

## Verdict

**FAIL** — REINFORCE is the bottleneck even at N=16 (not only a scale issue).  
Proceed automatically to **Stage C (PPO + GAE)**.

Artifacts: `metrics.json`, `plots.png`, `../n16_reinforce_train.log`
