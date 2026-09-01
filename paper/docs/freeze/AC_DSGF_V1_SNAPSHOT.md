# AC-DSGF v1 — Freeze Snapshot

**Status:** FROZEN for submission · 2026-07-17  
**Supersedes:** prior “baseline for ++” framing — ++ is supplement-only.

## Locked story
> AC-DSGF = learnable communication topology under an explicit budget,  
> maintaining cooperative task performance with far lower communication cost.

## Do NOT modify (v1)
| Asset | Path |
|-------|------|
| Encoder | `models/ac_dsgf.py` |
| Gate | `models/communication/controller.py` |
| Budget (hard) | `models/communication/budget_layer.py` |
| Config 16UAV | `configs/ac_dsgf/ac_dsgf_16uav.yaml` |
| Smoke | `configs/ac_dsgf/ac_dsgf_smoke_v0.yaml` |
| Claim lock | `paper/docs/freeze/AC_DSGF_FREEZE.md` |
| **Final freeze** | `paper/docs/freeze/AC_DSGF_FINAL_FREEZE.md` |

## Paper baseline (Table I, 16 UAV × 5 seeds)
Source: `paper/tables/table1_final.csv`

| Method | Success | Comm | CEI |
|--------|--------:|-----:|----:|
| DSGF | 4.01±1.68% | 39.27 | 0.0010 |
| **AC-DSGF (v1)** | **3.95±0.83%** | **0.43** | **0.0913** |

## Evidence artifacts
| Experiment | Artifact |
|------------|----------|
| Table I | `paper/tables/table1_final.csv` |
| Ablation | `paper/tables/table2_ablation.csv` |
| Generalization | `paper/tables/table3_generalization.csv` |
| Budget | `paper/tables/table_budget_sweep16.csv`, `fig_comm_budget16.png` |
| Packet loss | `paper/tables/table_packet_loss.csv` |
| Trigger | `paper/tables/table_comm_trigger_corr.csv` |
| PDF draft | `paper/ac_dsgf/` |
| Demo | `demo/videos/ac_dsgf_dynamic_comm.mp4` |

## AC-DSGF++ (not main)
See `paper/docs/freeze/SUPPLEMENT_S5_CAUSAL_UTILITY.md`.  
Code under `models/ac_dsgf_pp.py` / `configs/ac_dsgf_pp/` retained for reproducibility of the negative result only.
