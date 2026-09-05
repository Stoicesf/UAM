# AC-DSGF Swarm (ICPS + DICE + SEM)

Interactive multi-UAV stack: **ICPS** resource/channel layer → **DICE** roles & local rules → **SEM** semantic L1/L2/L3 gating.

![Demo](experiment_results/demos/demo_mixed.gif)

```
ICPS (bandwidth / SNR / compute)
        ↓
DICE  (roles · assignment · safety)
        ↓
SEM   (encoder · gate · L1/L2/L3 links)
        ↓
VMAS-style DICEVMASEnv  +  demos/visualizer
```

## Quick start (≈5 min)

```bash
# 1) env (locked GPU torch stack)
conda env create -f environment_frozen.yml   # or: conda activate pytorch12
pip install -e .

# 2) interactive demo
python demos/interactive_demo.py --scene search --n_agents 16 --speed 1x
# after install:
ac-dsgf-demo --scene tracking --n_agents 16
```

**Windows:** `.\scripts\run_demo.ps1 search 16 1x`  
**Linux/Mac:** `./scripts/run_demo.sh search 16 1x`

### Keyboard

| Key | Action |
|-----|--------|
| Space | Pause / resume |
| R | Reset episode |
| H | Help overlay |
| Q | Quit |

### Scenes

| Scene | What you see |
|-------|----------------|
| `search` | Point coverage / disperse to stars |
| `tracking` | Dynamic targets drift |
| `adversarial` | Mid-episode node kills + reassignment |
| `mixed` | Obstacles + light failures |
| `pursuit` | 12 UAVs encircle a fleeing evader |

异构可视化（`--hetero`）由环境初始化时的**先验引导的异构角色分配**驱动（机型比例 + 角色偏置），不加载 `role_policy.pt`。

Sim-to-real knobs: `--sensor_noise 0.05 --drop_rate 0.05 --comm_delay 3 --max_acc 2.0`

## Layout

| Path | Role |
|------|------|
| `models/complete_controller.py` | Unified ICPS+DICE+SEM step |
| `models/communication/` | Semantic gate + delay buffer |
| `dice/` | Roles, safety, failure injection |
| `environments/dice_vmas_env.py` | Torch swarm env (+ noise / dynamics / pursuit) |
| `demos/` | Interactive viz + scene library + recorder |
| `configs/stable/` | Frozen reproducible experiment configs |
| `scripts/` | Robustness / FPS / pursuit sweeps |

## Reproduce key experiments

```bash
python scripts/diagnose_semantic_gate.py --episodes 5
python scripts/robustness_sweep.py --episodes 3 --steps 40
python scripts/pursuit_generalization.py --seeds 10 --steps 100
python scripts/benchmark_demo_fps.py --steps 30
python demos/record_video.py --scene mixed --fps 8 --max_frames 40
```

Outputs land under `experiment_results/`.

## Install (editable)

```bash
pip install -e .
ac-dsgf-demo --scene pursuit
```

Requires: Python ≥3.10, `torch`, `numpy`, `matplotlib`, `pyyaml`, `vmas` (see `environment_frozen.yml` for pinned CUDA wheels).

---

## Archived research

Post-`v2.0-swarm-complete` emergence / hierarchical-PPO experiments are **not merged** into this branch. See the appendix in [`RELEASE_NOTES.md`](RELEASE_NOTES.md) (*已归档研究记录：角色涌现与覆盖-碰撞帕累托平衡*) and tag `archived/emergence-ppo` on `exp/hierarchical-ppo`.

Legacy DSGF-HRL paper-driven notes remain in git history / `paper/`; the live product surface is the demo stack above.

## Advanced work packages

```bash
# WP4 zero-shot transfer matrix
python scripts/transfer_matrix.py --episodes 20

# WP2 adversarial RL evader (smoke steps; bump for real training)
python scripts/train_adversarial.py --rounds 3 --evader_steps 50000 --hunter_steps 20000
python demos/interactive_demo.py --scene adversarial_pursuit --evader_policy rl

# WP3 click-to-explain
python demos/interactive_demo.py --scene adversarial --explain

# WP5 lifelong evolution
python demos/lifelong_demo.py --steps 5000 --save_plot
```
