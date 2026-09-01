# AC-DSGF Phase 1 Status (updated)

## Agreed narrative
Communication ↓ 90% → Success drop only ~11% (budget sweep 16UAV).
AC-DSGF = DSGF backbone + adaptive communication policy.

## Done
| Item | Artifact |
|------|----------|
| Table I++ (5 seeds, CEI) | `paper/tables/table1_plus_ac.csv`, `table1_plus_ac_paper.csv` |
| Budget sweep 16UAV | `paper/figures/fig_comm_budget16.png` |
| CEI = S/(C+ε), ε=1e-6 | MAPPO CEI = n/a (no graph) |

### Table I++ snapshot
| Method | Success | Comm | CEI |
|--------|---------|------|-----|
| MAPPO | 2.40±0.90% | 0 | — |
| GAT | 2.04±0.90% | 38.80 | 0.0005 |
| Transformer | 0.70±0.16% | 240 | ~0 |
| DSGF | **4.01±1.68%** | 39.27 | 0.0010 |
| **AC-DSGF** | **3.95±0.83%** | **0.43** | **0.0913** |

## In progress
Silence Collapse ablation (`scripts/eval_comm_ablation.py`):
- AC-full / AC-no_budget / AC-random on frozen 16UAV ckpt

## Next (locked order)
1. ~~Main table 5 seeds~~ → done (CEI finalized)
2. Communication ablation → running
3. Dynamic communication Demo
4. Method + Experiment writing
5. Submission polish

Do **not** retune λ / DSGF backbone.
