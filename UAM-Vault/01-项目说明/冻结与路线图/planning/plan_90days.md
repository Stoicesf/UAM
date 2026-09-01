# 90-Day Paper Plan — DSGF v2 (frozen, no more tuning)

## Current status (2026-07-12)
- DSGF v2 frozen: `configs/dsgf/dsgf_v2_frozen.yaml`
- Scalability 4/8/16/32 UAV: DONE
- Baseline16 Table I (seed=42): DONE
- **P0 in progress**: 5-seed × 4 methods (`scripts/run_5seed_baseline16.py`)
- **P0 pending**: AULC + t-test (`scripts/analyze_baseline16_stats.py`)

## Phase 1 — Month 1 (experiments)

| Week | Task | Status |
|------|------|--------|
| W1 | Scalability 4/8/16/32 UAV | DONE |
| W2 | Baselines @ 16 UAV (MAPPO/GAT/Transformer/DSGF) | DONE |
| W3-4 | 5-seed stability + AULC + significance | IN PROGRESS |

## Phase 2 — Month 2

| Week | Task |
|------|------|
| W5 | Communication radius ablation |
| W6 | Environment generalization |
| W7 | Extended ablation (sparse, adaptive beta) |
| W8 | Method math freeze |

## Phase 3 — Month 3

| Week | Task |
|------|------|
| W9 | Paper draft |
| W10 | Supplementary videos |
| W11 | Figures + Algorithm 1 |
| W12 | Submission prep |

## Execution
```bat
run_train.bat --exp configs/scalability/uav8.yaml --gate 3 --run-name uav8
python scripts/run_scalability.py
python scripts/plot_scalability.py
```

# Do NOT modify DSGF v2 algorithm after freeze.

## Scalability narrative (paper)
> DSGF maintains effective decentralized coordination under increasing swarm scale with sparse communication complexity.

Prove:
1. Success degrades slowly as N increases
2. Communication cost grows as O(kN) not O(N^2)
3. No centralized global state at execution

## Metrics saved per run
- `success_curve.npy`, `collision_curve.npy`, `reward_curve.npy`
- `communication.csv` with `sparse_edges`, `full_graph_edges`
- `summary.json` with `communication_cost_mean`

## Week 2 ready (after scalability)
```bash
python scripts/run_baseline16.py
```
Configs: `configs/baseline16/` — MAPPO, GAT, Full Attention, DSGF v2 @ 16 UAV

## Seed policy
- Scalability: seed=42 only (capacity eval with best config)
- Week 3: 5 seeds + AULC for final performance table
