# P1-3 Communication Efficiency Setup

## Goal

Verify that DSGF maintains competitive success under restricted communication budgets,
forming a Success–Communication Pareto curve (Table IV + Figure 6).

## Fixed Settings

| Parameter | Value |
|-----------|-------|
| UAV | 16 |
| seed | 42 |
| frames | 102,400 |
| scenario | navigation |
| algorithms | frozen (baseline16 hyperparams) |

## Variable: Communication Radius

| Key | `comm_radius` | Label |
|-----|---------------|-------|
| R0 | 0.0 | no edges |
| R1 | 1.0 | sparse |
| R2 | 2.0 | medium |
| R3 | 3.0 | dense |
| full | 100.0 | all pairs |

Both `env.comm_radius` and `guidance.comm_radius` are set together.

## Experiment Matrix

### Profile `full` (11 runs)

| Method | Radii |
|--------|-------|
| MAPPO | 0, 2, full |
| GAT | 0, 1, 2, full |
| DSGF | 0, 1, 2, full |

### Profile `fast` (9 runs, recommended)

All three methods × R ∈ {1, 2, full}.

## Metrics

- **Success** `S`: post-training eval (200 episodes)
- **Communication Cost** `C`: mean directed edges per timestep  
  `C = (1/T) Σ_t |E_t|`, logged in `communication.csv` during training
- **Efficiency** `η = S / C` (MAPPO: C=0 by definition)

## Commands

```bash
# Day 1 — smoke
python scripts/smoke_communication.py
python scripts/run_communication_train.py --profile smoke --gate 2

# Day 2-3 — formal training (fast = 9 runs)
python scripts/run_communication_train.py --profile fast --gate 3

# Day 4 — aggregate + Pareto
python scripts/eval_communication.py --profile fast --plot
```

## Outputs

```
results/communication/
├── mappo_r1/
├── gat_r2/
├── dsgf_rfull/
└── summary.csv

paper/tables/table4_communication.csv
paper/figures/fig6_communication_pareto.png
```

## Paper Narrative (safe wording)

> DSGF maintains competitive success under restricted communication budgets,
> demonstrating robustness to sparse connectivity.

Do **not** claim "DSGF always lower comm cost than GAT" unless data supports it.

## Note on MAPPO

MAPPO does not use explicit communication; success is radius-invariant.
Only one MAPPO run is needed for Pareto, but we train per-radius for table consistency
(comm_cost = 0 for all MAPPO rows).

## Note on Retraining

GAT/DSGF **must retrain** at each radius — graph structure is part of the forward pass.
Do not reuse R=0.5 checkpoints from Table I.
