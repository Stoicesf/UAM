# P1-1 Mechanism Validation — Ablation (4 UAV, seed=42, 102k)

## Key finding

Removing **any** DSGF component collapses success relative to Full model.
**Residual policy** is the dominant contributor (~42× success drop).

| Variant | Success | vs Full |
|---------|---------|---------|
| w/o Dynamic Graph | 0.28% | 33× lower |
| w/o Temporal | 0.86% | 11× lower |
| w/o Residual | 0.22% | 42× lower |
| **Full DSGF** | **9.27%** | — |

## Paper narrative (Section 4.4)

> Component ablation confirms that DSGF coordination gains arise from the
> joint effect of dynamic graph construction, temporal memory, and residual
> policy decoupling. Disabling residual injection causes the largest degradation,
> supporting our claim that decoupling learned guidance from local control mitigates
> long-horizon coordination collapse.

## Note on Table I vs Ablation

- **Table I (5-seed, 16 UAV)**: cross-method comparison under scale stress
- **Table II / Fig 4 (4 UAV ablation)**: internal mechanism validation — different setting, do not mix numerically in prose without stating setup

## Artifacts

- Figure: `paper/figures/fig4_ablation_components.png`
- Table: `paper/tables/table2_ablation.csv`
