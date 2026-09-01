# §6.6 Baseline Comparison Evidence

**Status:** Phase A **FROZEN** (corrected \(C\)) · Class C deferred  
**Protocol:** [`../tro_6_6_formal_protocol.md`](../tro_6_6_formal_protocol.md)

## \(C\) lock

| Method | \(C\) |
|--------|-------|
| MAPPO | \(0\) |
| GAT / DSGF | radius-neighborhood edges |
| Full Attention | dense \(N(N-1)=240\) |
| AC-DSGF | topology \(C_t\) |

## Key reading

AC-DSGF \(K=2\): \(J\approx9.24\), \(C\approx27.5\) — comparable return to Class B at lower cost.

## Files

| File | Content |
|------|---------|
| `baseline_comparison.csv` | per-seed rows |
| `evidence_6_6_report.json` | Table II aggregate |
| `collect_log_Cfix.txt` | corrected reeval log |
| `figures/Fig5_baseline_pareto.png` | \((C,J)\) plane |
