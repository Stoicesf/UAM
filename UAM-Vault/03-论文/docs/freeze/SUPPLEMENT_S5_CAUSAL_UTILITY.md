# Supplement S5 — Exploration of Causal Utility-driven Communication Selection

**Status:** Exploratory / negative result · **NOT** a main-paper contribution  
**Date:** 2026-07-17  
**Code branch:** `models/ac_dsgf_pp.py`, `configs/ac_dsgf_pp/` (do not merge claims into v1)

---

## S5.1 Motivation

Although adaptive topology (AC-DSGF) reduces unnecessary communication, an open question remains:

> Can agents further *predict the task influence of each message* and send only high-value links?

This motivates a preliminary extension **AC-DSGF++** (causal / outcome-aware utility).

---

## S5.2 Method (sketch)

Utility head:
\[
U_{ij}=f(h_i,h_j)\in(0,1)
\]
Gate conditioned on \(U\) (stop-grad into topology), with optional soft budget \(L_b=(C/C_t-1)^2\).

Counterfactual targets explored:
- **Action influence:** \(U^*=\|a^m-a^0\|\)
- **Outcome-aware (CAU):** mix of \(\Delta R\), \(\Delta V\), \(\Delta A\)

---

## S5.3 Failure Analysis — Utility Supervision Difficulty

4-UAV smoke lineage (seed 42, 20k; comparable setup):

| Version | Core change | Success | Comm | corr\((U,U^*)\) |
|---------|-------------|--------:|-----:|----------------:|
| v2c (action U*) | baseline ++ | 21.9% | 0.13 | 0.49 |
| v2e-B | soft budget | 15.1% | 0.97 | 0.33 |
| Detach backbone | \(U=f(h.\mathrm{detach()})\) | 15.0% | 0.97 | 0.46 |
| Anneal | curriculum budget | 12.5% | 1.03 | 0.42 |
| CAU-v1 | positive-clamp \(\Delta R/\Delta V\) | 8.3% | 0.94 | −0.02 |
| CAU-v2 | signed+batch-norm + H-step \(\Delta V\) | 6.0% | 0.83 | −0.09 |

### What was ruled out
| Hypothesis | Verdict |
|------------|---------|
| Hinge ranking broken | Ruled out (restored corr) |
| U interferes with policy | Ruled out (detach: Success flat) |
| Budget too tight / mistimed | Ruled out (relax + anneal insufficient) |
| **U\* label / long-horizon credit** | **Supported as primary bottleneck** |

### Interpretation (for supplement text)

> Accurate estimation of communication utility requires long-horizon causal credit assignment, which remains challenging under sparse-reward cooperative navigation. Immediate action change \(\|a^m-a^0\|\) is learnable but misaligned with task value; outcome-based labels either collapse or fail to stabilize policy success under our smoke protocol.

### 16-UAV Phase 6 (action-U*, 5 seeds) — not promoted to main
| Metric | AC-DSGF | AC-DSGF++ (Phase 6) |
|--------|--------:|--------------------:|
| Success | 3.95±0.83% | 3.38±0.85% |
| Comm | 0.43±0.09 | 2.56±1.05 |
| CEI | 0.093 | 0.020 |
| corr | — | 0.34±0.17 |

→ Does **not** support “worth sending” claim at formal scale.

---

## S5.4 Positioning in submission

- Main paper: **AC-DSGF only** (C1–C3 above).
- Supplement: this section as *preliminary exploration* + *limitation / future work*.
- Future direction: long-horizon outcome-aware communication value estimation — not claimed solved.

---

## Artifacts (local runs; not Table I)
| Run | Path |
|-----|------|
| v2c | `results/ac_dsgf_pp/ac_dsgf_pp_smoke_v2c/` |
| v2e-B | `results/ac_dsgf_pp/ac_dsgf_pp_smoke_v2e_b/` |
| Detach | `results/ac_dsgf_pp/ac_dsgf_pp_ablation_detach_u/` |
| Anneal | `results/ac_dsgf_pp/ac_dsgf_pp_anneal_smoke/` |
| CAU-v1 | `results/ac_dsgf_pp/ac_dsgf_pp_cau_v1_smoke/` |
| CAU-v2 | `results/ac_dsgf_pp/ac_dsgf_pp_cau_v2_smoke/` |
| Phase 6 16UAV | `results/ac_dsgf_pp/uav16/`, `paper/tables/table_pp16_*.csv` |
