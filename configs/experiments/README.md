# Thesis experiment roadmap (12-16 weeks)
# Each stage maps to one thesis subsection.

| Stage | Config | Paper | Compare |
|-------|--------|-------|---------|
| 0 Baseline | exp0_baseline.yaml | Ch5 Baseline | MAPPO |
| 1 Guide MLP | exp1_guide.yaml | Ch4.2 | MAPPO vs Guide |
| 2 Sparse Attn | exp2_sparse.yaml | Ch4.3 | Guide vs Sparse |
| 3 Dynamic Graph | exp3_graph.yaml | Ch4.4 | Sparse vs Graph |
| 4 Full DSGF | exp4_dsfg.yaml | Ch4.5 | Graph vs DSGF |
| 5 Ablation | experiments/ablation.py | Ch5.3 | remove components |
| 6 Scalability | exp6_scalability.yaml | Ch5.4 | 4/8/16/32 agents |
| 7 Communication | exp7_communication.yaml | Ch5.5 | comm cost |

## Commands

```bash
# Stage 0 — baseline (frozen, do not change)
python train.py --exp configs/experiments/exp0_baseline.yaml

# Stage 1 — Guide MLP (CURRENT)
python train.py --exp configs/experiments/exp1_guide.yaml

# Smoke test
python train.py --exp configs/experiments/exp1_guide.yaml --smoke

# Plot all results
python scripts/plot_results.py
```

## Each run saves

```
results/<run_name>/
  config.yaml          # original config copy
  config_resolved.yaml # merged config
  meta.json            # id, stage, seed
  metrics.csv          # step-wise metrics
  summary.json         # final curve + stats
  checkpoints/
  tensorboard/
```
