# AC-DSGF v1 Frozen Results Pointer

**2026-07-17:** Main submission path. AC-DSGF++ runs under `results/ac_dsgf_pp/` are **supplement / negative-result only** — do not overwrite v1 paths.

## Canonical tables
- `paper/tables/table1_final.csv`
- `paper/tables/table2_ablation.csv`
- `paper/tables/table3_generalization.csv`
- `paper/tables/table_budget_sweep16.csv`
- `paper/tables/table_packet_loss.csv`
- `paper/tables/table_comm_ablation.csv`
- `paper/tables/table_comm_trigger_corr.csv`
- `paper/tables/table5_compute_cost.csv`

## Expected run directories (if present)
```
results/ac_dsgf/uav16/s*/
results/ac_dsgf/ac_dsgf_smoke_v0/
```

## Paper package
```
paper/ac_dsgf/                    # IEEE draft (v1)
paper/ac_dsgf_final/              # submission pack snapshot
paper/docs/freeze/AC_DSGF_FINAL_FREEZE.md
paper/docs/freeze/SUPPLEMENT_S5_CAUSAL_UTILITY.md
```

## Rule
- **Do not** retrain or retune v1 for “better” Success.
- **Do not** promote `results/ac_dsgf_pp/` into main tables.
