# AC-DSGF Final Freeze Decision
# Date: 2026-07-17
# Status: **LOCKED — submit v1; ++ → supplement only**

## Decision
After a complete AC-DSGF++ exploration (Phase 5–6: budget, detach, anneal, CAU-v1/v2):

| Path | Action |
|------|--------|
| **AC-DSGF v1** | **Main paper — freeze & submit** |
| **AC-DSGF++** | **Supplement S5 — negative / exploratory result** |

**Stop** all ++ λ / budget / ranking / U* search.  
**99% effort → v1 writing & packaging; 1% → ++ as future work.**

---

## Locked title
**Adaptive Communication-Constrained DSGF for Efficient UAV Swarm Coordination**

## Locked research question
> Under communication resource constraints, how can a UAV swarm learn an *effective communication topology* rather than exchanging unbounded messages?

## Locked claim
> AC-DSGF achieves communication-efficient cooperative navigation while maintaining task performance comparable to dense-graph DSGF.

### Do NOT claim (main paper)
- Causal / outcome-aware communication as a verified contribution
- Success leadership vs DSGF
- AC-DSGF++ as proposed method

---

## Contributions C1–C3 (main paper only)

### C1 — Learnable Communication Topology
\[
g_{ij}=\sigma\bigl(W[h_i,h_j,d_{ij},\rho]\bigr)
\]
vs fixed radius \(A_{ij}=\mathbf{1}(d_{ij}<r)\) or random dropout.  
**Evidence:** Silence-collapse ablation · Dynamic-comm demo · Trigger analysis

### C2 — Communication Budget Optimization
\[
\max\; R-\lambda C,\qquad C=\sum_{ij} g_{ij}
\]
**Evidence:** Budget sweep · Pareto · CEI (\(S/(C+\varepsilon)\))

### C3 — Residual Spatial-Temporal Guidance
\[
a = a_{\mathrm{local}}+\Delta a_{\mathrm{guide}}
\]
**Evidence:** Ablation w/o residual · Table I DSGF backbone

---

## Table I (frozen, 16 UAV × 5 seeds)
Source: `paper/tables/table1_final.csv`

| Method | Success | Comm | CEI |
|--------|--------:|-----:|----:|
| GAT | 2.04±0.90% | 38.80 | 0.0005 |
| DSGF | **4.01±1.68%** | 39.27 | 0.0010 |
| **AC-DSGF** | **3.95±0.83%** | **0.43** | **0.0913** |

Story: Success ≈ DSGF · Comm ↓ ~90× · CEI ↑ ~90×

---

## Main experiment map (do not expand)

| Exp | Content | Artifact |
|-----|---------|----------|
| I | Overall 16UAV×5seed | `table1_final.csv` |
| II | Ablation (DG / Temporal / Residual) | `table2_ablation.csv` |
| III | Budget Pareto | `table_budget_sweep16.csv`, `fig_comm_budget16.png` |
| IV | Generalization (obstacles) | `table3_generalization.csv` |
| V | Packet loss | `table_packet_loss.csv` |
| VI | Behavior / triggers | `table_comm_trigger_corr.csv` |

Demo narrative: *task completion with substantially fewer interactions* — not “highest success”.

---

## Forbidden (post-freeze)
- Retrain / retune v1 λ or architecture
- Promote ++ numbers into main tables
- Chase Success↑ as primary story
- New modules (VQ-VAE, world models, physical channel)

## Allowed
- Writing, captions, compile polish
- Cover letter / mock reviewer response
- Optional presentation-only demos (no claim change)
