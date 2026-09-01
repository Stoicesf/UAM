# Results Database Structure

```
results/
├── baseline/          # Stage 0 — frozen MAPPO
│   └── <run_name>/
├── guide/             # Stage 1 — Guide MLP (Guide_v1)
│   └── guide_gate3_v1/
├── sparse/            # Stage 2
├── graph/             # Stage 3
└── dsgf/              # Stage 4 Full DSGF
```

Each run contains:

```
<run_name>/
├── config.yaml
├── meta.json
├── summary.json       # paper metrics: reward, success, collision, alignment, path_length
├── TAG.txt            # e.g. Guide_v1 (frozen tag)
├── metrics.csv
├── action_alignment.csv
├── communication.csv
├── checkpoints/
│   ├── checkpoint_20k.pt
│   ├── checkpoint_40k.pt
│   ├── ...
│   ├── latest.pt
│   └── final.pt
├── plots/             # Figure 1-4 auto-generated
└── tensorboard/
```

## 5-seed baseline16 (P0)

```bash
# Train 5 seeds × 4 methods (skips existing)
python scripts/run_5seed_baseline16.py --seeds 3407,2026,1234,8888

# Auto-watch + post-pipeline (stats, figures, tables)
python scripts/watch_and_run_pipeline.py
```

Post-pipeline (`pipeline_after_5seed.py`) runs automatically when 20 summaries exist:
- `analyze_baseline16_stats.py` — mean±std, 95% CI, AULC, t-test
- `plot_baseline16_5seed.py` — learning curve + success/AULC bars
- `paper/tables/table1_baseline_5seed.csv`
- `results/pipeline_status.json` — next-task checklist (P1/P2)

## Gate 3 (running now)

```bash
python train.py --exp configs/experiments/exp1_guide.yaml --gate 3 --run-name guide_gate3_v1
```

Output: `results/guide/guide_gate3_v1/`

## After training

```bash
# Generate Figure 1-4
python scripts/generate_stage1_figures.py --guide-run results/guide/guide_gate3_v1

# With baseline comparison (when available)
python scripts/generate_stage1_figures.py \
  --guide-run results/guide/guide_gate3_v1 \
  --baseline-run results/baseline/<baseline_run>
```

## Paper figures roadmap

| Figure | Script | Stage |
|--------|--------|-------|
| 1-4 | generate_stage1_figures.py | Stage 1 |
| 5-6 | TBD | Stage 2-3 |
| 7-9 | plot_results.py | Stage 5-7 |
