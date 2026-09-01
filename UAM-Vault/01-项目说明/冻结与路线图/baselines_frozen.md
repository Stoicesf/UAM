# Frozen Baselines — do not tune further

## GAT-MAPPO (B3)
- Config: `configs/gat/gat_a_lambda003.yaml`
- Run: `results/graph/gat_a_lambda003/` (20k, success=4.74%)
- Role: graph coordination baseline + training-degradation evidence (compare with 102k)

## MLP Guide (B2)
- Config: `configs/experiments/ablation_c_warmup.yaml`
- Best 10k: `results/guide/ablation_c_warmup/` (success=2.54%)

## MAPPO (B1)
- Config: `configs/experiments/exp0_baseline.yaml`

## DSGF (Ours)
- Target config: `configs/experiments/exp4_dsfg.yaml` (to be updated after Week 2 implementation)
