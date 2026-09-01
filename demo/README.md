# Paper-level Demo

Frozen-checkpoint rollouts for Supplementary Video / defense. **Not** Table metrics.

## Comparison (recommended)

| Method | Checkpoint | Demo best S (8 eps) | Video |
|--------|------------|---------------------|-------|
| MAPPO | `results/demo/mappo_4uav` (~20k) | 50% | `videos/mappo_4uav.mp4` |
| GAT-A | `results/graph/gat_a_lambda003` (20k) | 50% | `videos/gat_4uav.mp4` |
| DSGF | `results/dsgf/dsfg_v2_102k` **checkpoint_20k** | **75%** | `videos/dsgf_4uav.mp4` |

Side-by-side:

```
demo/figures/comparison_trajectories.png
demo/videos/comparison_mappo_gat_dsgf.mp4
```

## Commands

```bash
# Individual
python scripts/demo_runner.py --method dsgf --ckpt-prefer 20k --episodes 8 --select-best
python scripts/demo_runner.py --method gat  --ckpt-prefer 20k --episodes 8 --select-best
python scripts/demo_runner.py --method mappo --ckpt-prefer final --episodes 8 --select-best

# Panel + strip video
python scripts/demo_comparison.py --methods mappo gat dsgf
```

## Why checkpoint_20k for DSGF/GAT

Task success peaks early; long training can raise reward while lowering eval success.
Demo selects the visually best episode among several rollouts — for illustration only.

## Paper wording

> Supplementary Video: Cooperative UAV Navigation with Dynamic Sparse Graph Fusion  
> We visualize policies selected by task success (early checkpoint / best demo rollout), not final training reward.
