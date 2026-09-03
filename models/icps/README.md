# ICPS Parallel Track — Branch Isolation

**Branch**: `feature/icps-phase1`  
**Status**: ACTIVE (parallel to Paper Landing)  
**Rule**: Do not merge into paper-freeze / main until Phase-6 submission package is green.

## Isolation

| Path | Role |
|------|------|
| `models/icps/` | All new ICPS code |
| `configs/icps/` | Channel / gate defaults |
| `experiments/icps/` | Dual-gate, compute Pareto, ICPS-07 |
| `scripts/icps/` | Self-checks + AAS SNR calibration |
| `env/icps_scenarios.py` | Scene knobs |
| `UAM-Vault/06-ICPS知识库/` | Literature notes |
| `paper/docs/research/ICPS_*.md` | Roadmap / feasibility / memo |

## Default-off switches

- `DynamicGraphModule(quality_source="geometric")` — mainline default
- Set `quality_source="icps_channel"` only in ICPS experiments
- `configs/icps/icps_defaults.yaml` → `enabled: false`

## Forbidden on this track

- Editing frozen ++ / causal / secdo modules for ICPS features
- Changing mainline λ / Success-chase claims
- Claiming AAS as algorithmic baseline
