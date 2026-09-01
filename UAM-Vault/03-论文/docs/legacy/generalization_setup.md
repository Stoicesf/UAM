# P1-2 Generalization Setup

## Step 1 — VMAS Navigation obstacle audit (2026-07-13)

**Finding:** `vmas/scenarios/navigation.py` has **no** `n_obstacles` / static obstacle support.
Only agent goals (`Landmark`, non-collidable) and agent-agent Lidar.

**Action:** Added project scenario `env/scenarios/navigation_obstacle.py`
- Parameter: `n_obstacles` (via `env.num_obstacles` or `scenario_kwargs`)
- Static red `Landmark` obstacles, collidable
- Obstacle-agent collision penalty in reward
- **Observation dim unchanged (18 @ 4 agents)** when lidar rays = 12
- Lidar detects agents + obstacles when `n_obstacles > 0`

## Step 2 — Configs

```
configs/generalization/
├── train_env.yaml
├── test_obs{0,2,4,6,8}.yaml
├── manifest.json
└── methods/
    ├── mappo_obs4.yaml
    ├── gat_obs4.yaml
    └── dsgf_obs4.yaml
```

## Step 3 — Training (NOT started)

Train once on `obs=4`:
```bash
python train.py --exp configs/generalization/methods/mappo_obs4.yaml --gate 3 --run-name mappo_obs4
python train.py --exp configs/generalization/methods/gat_obs4.yaml --gate 3 --run-name gat_obs4
python train.py --exp configs/generalization/methods/dsgf_obs4.yaml --gate 3 --run-name dsgf_obs4
```

## Step 4 — Zero-shot eval pipeline

```bash
python scripts/eval_generalization.py --plot
```

- Loads checkpoint, `model.eval()`, `torch.no_grad()`
- No `optimizer.step()`
- Metrics: success, collision, reward, path_length, **generalization gap G**
- Outputs: `paper/tables/table3_generalization.csv`, `fig5_generalization.png`

Smoke test (plain-navigation fallback ckpt, distribution shift only):
```bash
python scripts/eval_generalization.py --allow-fallback --plot
```

## Important

Existing 4-UAV checkpoints (`dsfg_v2_102k`, etc.) were trained on **plain navigation**
without obstacles. Meaningful Table III requires **re-training on obs=4** first.
