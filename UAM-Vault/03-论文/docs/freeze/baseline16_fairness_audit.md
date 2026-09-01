# Baseline16 Fairness Audit — 2026-07-12

All methods share identical environment and evaluation protocol.

| Parameter | Value |
|-----------|-------|
| scenario | navigation |
| num_agents | 16 |
| num_envs | 8 |
| max_steps | 128 |
| total_frames | 102400 |
| eval_episodes | 200 |
| seed | 42 |

## Observation fairness

- **MAPPO**: actor/critic input = VMAS local observation only (18-dim).
- **GAT / Transformer / DSGF**: same env observation; guidance Φ is learned from
  per-agent local obs (+ comm-limited neighbor aggregation). No global privileged state.

## Intentional method differences only

| Method | Guidance | Residual | Guide Reward |
|--------|----------|----------|--------------|
| MAPPO | none | — | no |
| GAT | static GAT | concat | yes |
| Transformer | dense attention | concat | no |
| DSGF | dynamic sparse + temporal | yes (β) | no |
