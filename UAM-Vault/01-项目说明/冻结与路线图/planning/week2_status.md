# Week 2 Environment & Experiment Status

## Entry test (non-blocking)

| Command | Status | Note |
|---------|--------|------|
| `conda run -n dpg_hrl python train.py` | FAILED | `UnicodeEncodeError` — GBK stdout |
| `E:\ANACONDA\envs\dpg_hrl\python.exe train.py` | OK | **Fixed execution path** |
| `run_train.bat --exp ...` | OK | UTF-8 wrapper for all args |

### Fixed paper environment

```
OS:         Windows 11
Python:     3.10 (conda dpg_hrl)
GPU:        RTX 4060 8GB, CUDA 12.1
PyTorch:    2.5.1+cu121
Simulator:  VMAS
Execution:  Direct Python interpreter (or run_train.bat)
```

---

## Week 2 validation (completed 2026-07-10)

### DSGF v2 full — 3 seeds @ 102k

| Run | Success | Collision | Reward |
|-----|---------|-----------|--------|
| s42 | **9.27%** | 0.147 | -6.13 |
| s3407 | 0.56% | — | -7.85 |
| s2026 | 0.74% | — | -6.82 |
| **mean** | **3.52%** | — | — |
| **std** | **4.85%** | — | — |

### Ablation @ 102k (seed=42)

| Method | Success |
|--------|---------|
| DSGF v2 full | **9.27%** |
| w/o Residual | 0.22% |
| w/o Dynamic Graph | 0.28% |
| w/o Temporal | 0.86% |

**Key finding:** Residual ablation drops 42x (9.27% vs 0.22%) on seed 42.

**Open issue:** High seed variance — need analysis in paper (report mean+-std, discuss stochasticity).

---

## Paper evidence chain

```
MAPPO baseline     -> OK
MLP Guide          -> reward misalignment discovered
GAT Guide          -> long-horizon degradation (-95%)
DSGF v1            -> partial fix (-88%)
DSGF v2 (frozen)   -> core result (-65%, residual key)
```

## Next priority

1. Document seed variance + optional seed 0 rerun
2. Phase 3: scalability 4/8/16/32 UAV
3. Figure 2: success vs training steps (20k/50k/102k curve)
