# Theory & Evidence Alignment — Revision Log (AC-DSGF v1)

**Goal:** Align theory claims with evidence. No algorithm expansion.

---

## Done (P0)

### Prop.1 → Communication scaling under degree constraint
- Equality \(\eta_N=K/(N-1)\) removed
- Now: \(|E_t(i)|\le K\) ⇒ \(\eta_N\le K/(N-1)=\mathcal{O}(1/N)\)
- Wording: inverse-order density decrease under bounded neighborhood (not “converges to zero” as a theorem slogan)

### Prop.2 → Residual Stability Analysis
- Removed “task preservation / performance non-decrease”
- Bound on \(\|a-a^\star\|\le\beta L_\pi\varepsilon\) only

### Prop.3 → Topology Optimality Interpretation (new)
- Gate as surrogate for \(\max\sum g_{ij}u_{ij}\) s.t. budget

### AC-random narrative
- Explicit: random may raise Success via noise reduction
- Does **not** match SCA efficiency of AC-full
- Hard Top-K added (Table 2b / Fig.14)

### Table 1 vs Table 2
- Protocol note: Table 1 = main eval; Table 2 = frozen-ckpt interventions (not interchangeable)

### SCA
- Soft Communication Activation defined once; used thereafter

### Algorithm 1
- Expanded to scoring → relaxation → budget → aggregate → residual

---

## Hard Top-K result (honest)

| Method | Success % (mean±std) | \|E\| |
|--------|---------------------:|------:|
| Random Top-2 | 25.1±13.0 | 26.7±2.0 |
| Distance Top-2 | 29.1±12.7 | 26.6±1.5 |
| AC Top-2 | 28.2±11.7 | 26.3±2.0 |

**Do not claim AC dominates Distance under hard K.** Claim: competitive under matched budget; learning value = soft SCA regime + edge importance.

Script: `scripts/eval_hard_topk_fixed.py`  
Table: `paper/tables/table_hard_topk_k2.csv`

---

## Remaining (P1, optional)

- Full figure renumber (Fig 0→14 continuous) — index updated; file names kept for compatibility
- Multi-seed mean±std on Table 1 if re-eval budget allows


---

## Final polish (2026-07-18 evening)

- Prop.3 softened to *Interpretation of Adaptive Topology Learning* (approximate / induced; not a solver claim)
- Hard Top-K repositioned: comparable ≠ outperform; SCA is the primary story
- Title → Learning Adaptive Topologies for Communication-Constrained UAV Swarm Coordination
- Intro → three-layer motivation (infrastructure / attention≠activation / when-and-with-whom)
- Contributions locked to RA-L style (joint opt / differentiable adaptation / SCA validation)
- Fig.13 redrawn as budget–performance operating curve (frozen policy). **Full multi-λ retrain sweep deferred** (would require 5 trainings); do not claim λ_c Pareto from eval-only interventions.

## Corollaries (2026-07-18)

- **Corollary 1:** Task-return degradation bound \(|J_{full}-J_{sparse}|\le L_r\beta L_\pi\varepsilon T\) (after Prop.2)
- **Corollary 2:** Top-K utility gap \(\le 2K\delta\) under bounded score error (after Prop.3)
- Still interpretive / conditional; no claim that sparsity improves return or that Top-K training is superior.
