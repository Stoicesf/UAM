# SECDO UAV Experiments (Phase 2B–2D)

**Goal:** Prevent reviewers from reducing SECDO to “forecast head + PGD.”  
**Algorithm:** [`../../paper/ac_dsgf_v2/algorithm_secdo.md`](../../paper/ac_dsgf_v2/algorithm_secdo.md)

---

## Layout

```text
experiments/secdo_uav/
  env_uav_bandwidth.py   # mobility + SINR teacher c_t
  metrics.py             # ε, δ, ρ, Gap, violation
  runner.py              # train predictors + eval methods
scripts/run_secdo_phase2.py
paper/ac_dsgf_v2/experiments/phase2_results/
```

---

## Regimes

| ID | Name | Intent |
|----|------|--------|
| E1 | `slow` | \(\rho\approx 0\); little room for anticipatory gain |
| E2 | `fast` | large \(\rho\); expect benefit when \(\delta<\rho\) |
| E3 | `stress` | inflate \(\epsilon,\delta\) via noise; Gap should rise |

---

## Methods / baselines

| Method | Role |
|--------|------|
| `reactive` | \(\Pi_{\mathcal{B}(c_t)}\) |
| `oracle` | \(\Pi_{\mathcal{B}(c_{t+1})}\) upper bound |
| `secdo` | \(\Pi_{\mathcal{B}(\hat c_{t+1})}\) |
| `no_latent` | Ablation A — kill dynamics forecast |
| `no_constraint_dyn` | Ablation B/C — no \(\hat c\); reactive set |

Paper table may still list DSGF / AC-DSGF as *topology-instantiation* baselines in a later coupling stage; Phase 2B first closes the **constraint-optimization** loop.

---

## Mandatory logs (Phase 2C)

- \(\epsilon_t\), \(\delta_t\), \(\rho_t\)
- \(\mathrm{frac}(\delta<\rho)\)
- \(\mathrm{Gap}\), violation
- Scatter: \(\frac1T\sum(\epsilon+\delta)\) vs \(\mathrm{Gap}\) → `theory_prediction_scatter.json`

---

## Run

```bash
E:\ANACONDA\envs\dpg_hrl\python.exe -u scripts/run_secdo_phase2.py
```
