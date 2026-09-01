# Exp-2 Communication Budget Sweep — Notes

**Date:** 2026-07-15  
**Script:** `scripts/eval_comm_budget.py`  
**Data:** `results/ac_dsgf/budget_sweep/budget_sweep.csv`  
**Figure:** `paper/figures/fig_comm_budget.png`

## Protocol

- Checkpoints: GAT-A 20k, DSGF v2 20k, AC-DSGF smoke 20k (4 UAV)
- Episodes: 32 per (method × budget)
- Budgets: 100% / 75% / 50% / 25% / 10% of in-radius neighbors (hard top-k)
- GAT/DSGF: top-k by distance / quality scores  
- AC-DSGF: top-k by learned \(g_{ij}\)
- **CE** = Success / CommCost

## Headline (4 UAV smoke scale)

| Method | Success range | Comm scale | CE scale |
|--------|---------------|------------|----------|
| GAT | 25–34% | ~1.9–2.3 | ~0.12–0.16 |
| DSGF | 48–52% | ~1.3–1.8 | ~0.27–0.40 |
| **AC-DSGF** | **48–55%** | **~0.06–0.08** | **~7–9** |

**Engineering takeaway (safe wording):**

> Under hard communication budgets, AC-DSGF maintains competitive task success while using substantially lower communication mass than radius/quality graph baselines, yielding much higher communication efficiency (CE).

## Caveats for paper writing

1. This run is **4 UAV / early checkpoints** — re-run at **16 UAV · 102k · multi-seed** before calling it Table I′.
2. AC CommCost is soft \(\sum g\); GAT/DSGF use hard edge counts after top-k — report both definitions transparently.
3. Do not claim “70% less communication with <10% drop” until 16 UAV replicates the pattern under matched cost units.

## Next

- Matched hard-threshold CE for AC (`g ≥ τ`) for fairer cost units  
- Scale Exp-2 to 16 UAV  
- Demo: dual-pane trajectory + live \(g_{ij}\) graph under mid-episode budget shock
