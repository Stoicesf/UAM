# AC-DSGF++ Phase 6–9 Execution Plan
# Status: Phase 5 v2c PASS → Phase 6 formal IN PROGRESS
# Updated: 2026-07-16

## Locked claim
> Performance Preservation + Communication Reduction  
> AC-DSGF++ learns whether communication is worth performing by estimating
> its influence on downstream decisions.

**Do NOT chase Success↑.** Target: Success ≈ AC-DSGF, Comm ↓, CEI ↑, corr(U,U*)>0.5

## Phase checklist

| Phase | Goal | Status |
|-------|------|--------|
| 5 | Smoke v2c | ✅ PASS |
| **6** | **16UAV × 5 seeds formal** | **▶ NOW** |
| 7 | Causal ablations (w/o U, random, distance) | configs ready |
| 8 | Demo: utility heatmap + SEND/DROP | ⬜ |
| 9 | Paper title + C1–C3 rewrite | ⬜ |

## Phase 6 commands

```bash
# All 5 seeds (Table I: 42,3407,2026,1234,8888)
python scripts/run_5seed_ac_dsgf_pp16.py

# Start with one seed
python scripts/run_5seed_ac_dsgf_pp16.py --seeds 42

# After all seeds
python scripts/analyze_ac_dsgf_pp16_table.py
```

Results: `results/ac_dsgf_pp/uav16/s{seed}/`

## New metrics (logged under `<run>/logs/`)
| Metric | File | Target |
|--------|------|--------|
| corr(U,U*) | utility_corr.csv | >0.5 late |
| Comm Precision | comm_precision.csv | ↑ vs random |
| Gate mass | gate_mass.csv | compress after warm-up |
| CEI | table | > AC-DSGF (~0.09) |

## Phase 7 ablation configs
- `configs/ac_dsgf_pp/ablation16_wo_utility.yaml`
- `configs/ac_dsgf_pp/ablation16_random_utility.yaml`
- `configs/ac_dsgf_pp/ablation16_distance_utility.yaml`

## Paper (Phase 9)
**Title:** Causal Action-Aware Communication Learning for Resource-Efficient UAV Swarms

- C1: Causal utility communication (action influence)
- C2: Utility-guided adaptive graph \(g=f(h,d,U)\)
- C3: Reduce *invalid* communication, not just count

## Forbidden
VQ-VAE · World Model · Physical topology · ROS · retune backbone / λ chase Success
