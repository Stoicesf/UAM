# AC-DSGF Paper Freeze — Contribution & Claim Lock
# Updated: 2026-07-17
# Status: **FINAL FREEZE — submit v1; see AC_DSGF_FINAL_FREEZE.md**

## Locked claim (English)
> AC-DSGF achieves communication-efficient cooperative navigation
> while maintaining comparable task performance.

## Locked claim (中文)
> AC-DSGF 在保持任务性能的同时，实现自适应通信控制，大幅降低通信开销。

## Do NOT claim
- “AC-DSGF improves / achieves higher success rate” as the primary message
- Absolute Success leadership vs DSGF
- Causal / outcome-aware communication (AC-DSGF++) as a main contribution

## Title (locked)
Adaptive Communication-Constrained DSGF for Efficient UAV Swarm Coordination

## Contributions (locked C1–C3)

### C1 — Learnable Communication Topology
\[ g_{ij}=\sigma(W[h_i,h_j,d_{ij},\rho]) \]
Evidence: Silence Collapse ablation + Dynamic Comm demo + Trigger analysis

### C2 — Communication Budget-aware Optimization
\[ J = R - \lambda_c C_{comm},\quad C_{comm}=\sum_{ij} g_{ij} \]
Evidence: 16UAV budget sweep · Pareto · CEI

### C3 — Residual Spatial-Temporal Guidance
\[ a = a_{local} + \Delta a_{guide} \]
Evidence: Ablation w/o residual · Table I DSGF backbone

## Algorithm freeze
- DSGF v2 backbone: `configs/dsgf/dsgf_v2_frozen.yaml` — DO NOT MODIFY
- AC-DSGF: `configs/ac_dsgf/ac_dsgf_16uav.yaml` — DO NOT tune λ / architecture
- Forbidden: retune λ, swap Transformer, add networks, chase max Success, promote ++ to main

## Table I final (16 UAV × 5 seeds × 102k)
Source: `paper/tables/table1_final.csv`
| Method | Success | Comm | CEI |
|--------|---------|------|-----|
| GAT | 2.04±0.90% | 38.80 | 0.0005 |
| DSGF | **4.01±1.68%** | 39.27 | 0.0010 |
| **AC-DSGF** | **3.95±0.83%** | **0.43** | **0.0913** |

Main story: Success ≈ DSGF, Comm ↓ ~90×, CEI ↑ ~90×

## AC-DSGF++
Supplement only → `SUPPLEMENT_S5_CAUSAL_UTILITY.md`

## CEI definition
\[ \mathrm{CEI}=\frac{S}{C+\varepsilon},\ \varepsilon=10^{-6} \]
