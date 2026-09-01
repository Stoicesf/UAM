# Stage 1 Gate Workflow

| Gate | Command | Purpose | Time |
|------|---------|---------|------|
| **1** | `python train.py --exp configs/experiments/exp1_guide.yaml --gate 1` | Forward + phi stats | ~5 min |
| **2** | `python train.py --exp configs/experiments/exp1_guide.yaml --gate 2 --run-name guide_gate2` | 10k frames, reward rising | ~10 min |
| **3** | `python train.py --exp configs/experiments/exp1_guide.yaml --gate 3 --run-name guide_gate3` | 102k formal run | ~15 min |
| **4** | `python scripts/run_seeds.py --gate 3` | 3 seeds Mean±Std | ~45 min |

## Checklist before Gate 3

- [ ] Gate 1: phi mean/std non-zero, no NaN
- [ ] Gate 2: `reward_goal`, `reward_guide` rising in TensorBoard
- [ ] Gate 2: `action_alignment.csv` alignment trending up
- [ ] Baseline frozen at `exp0_baseline.yaml`

## Logs per run

```
results/<run>/
  metrics.csv              # all scalars
  action_alignment.csv     # cos(theta_action - theta_guide)
  communication.csv        # placeholder for Stage 4+
  tensorboard/             # reward/goal, reward/collision, reward/guide
```

## Figure 1

After Gate 3 baseline + guide:
```bash
python scripts/plot_results.py
# -> figures/figure1_baseline_vs_guide.png
```
