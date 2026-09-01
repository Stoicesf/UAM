# P2 Paper Writing Status

**Phase:** Experiment freeze complete. Writing in progress. **Do not retrain / retune DSGF.**

## Frozen Assets

| Artifact | Path |
|----------|------|
| Freeze manifest | `paper/experiment_freeze.json` |
| Table I | `paper/tables/table1_main.csv` |
| Table II | `paper/tables/table2_ablation.csv` |
| Table III | `paper/tables/table3_generalization.csv` |
| Table IV | `paper/tables/table4_communication.csv` |
| Table V | `paper/tables/table5_complexity.tex` (+ `.csv`) |
| Fig 2–6 | `paper/figures/fig{2,3,4,5,6}_*.png` |

## LaTeX Chapters

| Section | File | Status |
|---------|------|--------|
| I Introduction | `chapter1_intro.tex` | Draft skeleton |
| III Method (+ Complexity) | `chapter3_method.tex` | IEEE draft complete |
| IV–V Setup / Results / Limitations | `chapter4_exp.tex` | IEEE draft complete |

## Safe Claims (use these)

1. **Main:** DSGF improves mean success vs MAPPO/GAT at 16 UAV (5-seed).
2. **Mechanism:** Residual decoupling is critical (Ablation 4 UAV).
3. **Communication:** Better under *restricted* $R_c$ (R=1/2); not full-graph domination.
4. **Complexity:** Attention $O(Nkd)$ vs dense $O(N^2 d)$ when $k\ll N$.

## Forbidden Claims

- Do not mix Table I (16 UAV) with Ablation (4 UAV) numbers.
- Do not claim “DSGF always lower communication cost than GAT”.
- Do not claim real-world flight readiness from VMAS success rates.

## Next Writing Tasks

1. Related Work section
2. Abstract + Conclusion
3. Single master `main.tex` assembling chapters + `\input` tables
4. Optional: AULC bar figure already exists as `fig_auc_baseline16.png` — cite in stability paragraph
