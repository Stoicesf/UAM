# DSGF v2 FROZEN — Paper Reference

**Frozen date:** 2026-07-10  
**Do not modify:** residual structure, beta decay, policy input, DSGF encoder topology.

## Canonical config
`configs/dsgf/dsgf_v2_frozen.yaml`

## Baseline run (seed=42, 102k)
`results/dsgf/dsgf_v2_102k/`

| Metric | 20k | 102k |
|--------|-----|------|
| Success | 26.27% | **9.27%** |
| Degradation | — | -64.7% (vs GAT -95%, v1 -88%) |

## Paper claim (Table 1 — Long-horizon stability)
Policy-level guidance decoupling (residual β decay, no guide reward) maintains task success under extended training while GAT/v1 collapse.

## Week 2 validation matrix
| Experiment | Config | Priority |
|------------|--------|----------|
| DSGF v2 full ×3 seeds | `dsgf_v2_frozen.yaml` | Reproducibility |
| w/o Residual | `ablation_wo_residual.yaml` | **Highest** |
| w/o Dynamic Graph | `ablation_wo_dynamic_graph.yaml` | C1 |
| w/o Temporal | `ablation_wo_temporal.yaml` | C2 |

Run: `python scripts/run_week2_validation.py`
