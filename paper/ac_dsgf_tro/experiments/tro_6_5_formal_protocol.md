# §6.5 Channel Robustness — Formal Protocol

**Status:** **FROZEN in manuscript** (EN/CN v0.9 §6.5)  
**Title (manuscript):** Robustness under Imperfect Communication Channels  
**Purpose:** Stress-test topology decisions under realistic channel impairments — not a new robustness theorem.  
**No training. No encoder / scorer / reward / budget-definition edits.** Frozen checkpoints only.

---

## Claim hygiene

Forbidden:
> AC-DSGF is robust to communication failures.

Required:
> AC-DSGF maintains more stable coordination performance under degraded communication conditions.

---

## Methods (three only)

| Tag | Method | Role |
|-----|--------|------|
| `ac_dsgf` | AC-DSGF (frozen `uav16`) | learned topology + \(\Pi_{B_t}\) |
| `dsgf` | DSGF | fixed / non-budgeted sparse graph family |
| `full_attn` | Full Attention | dense communication |

Do **not** re-benchmark the full §6.6 suite.

---

## Factors (frozen)

| Channel | Levels | Implementation |
|---------|--------|----------------|
| Packet loss \(p_\ell\) | \(\{0,0.1,0.3,0.5\}\) | i.i.d. Bernoulli drop on active edges / gates |
| Delay \(\tau\) | \(\{0,20,50,100\}\) ms → **sim steps** \(\{0,1,2,4\}\) (dt≈25 ms) | use \(G_{t-\tau}\) / stale gates for aggregation |
| Bandwidth \(B/B_{\mathrm{full}}\) | \(\{0.1,0.2,0.4,0.8\}\) | hard ratio / Top-\(K\) projection (AC-DSGF); matched sparsity proxy for others |

Sweep **one factor at a time** (others at nominal zero / full).

---

## Metrics

| Symbol | Definition |
|--------|------------|
| \(J\) | episode return |
| Success | goal reach rate |
| \(C\) | mean active communication cost |
| \(C_{\mathrm{eff}}\) | \((1-p_\ell)C\) under packet loss |
| \(\lvert E_t\rvert\) | mean hard edge count (topology adaptation) |
| \(\varepsilon_G\) | twin message discrepancy vs full-support reference (when available) |

---

## Eval settings

| Item | Spec |
|------|------|
| \(N\) | 16 |
| Seeds | \(\{1234,2026,3407,42,8888\}\) |
| Episodes / cell | **16** (feasibility of trends; not a bake-off) |
| `max_steps` | 128 |
| Device | cuda |

---

## Outputs

```text
paper/ac_dsgf_tro/experiments/evidence_6_5_channel/
├── README.md
├── channel_statistics.csv
├── evidence_6_5_report.json
└── figures/
paper/ac_dsgf_tro/figures/Fig6_channel_robustness.png   # (a) loss (b) delay (c) bandwidth
```

---

## Command

```bash
E:\ANACONDA\envs\dpg_hrl\python.exe -u scripts/collect_tro_channel_robustness.py `
  --seeds 1234 2026 3407 42 8888 `
  --episodes 16 `
  --device cuda
```

Smoke:
```bash
... --seeds 1234 --episodes 4 --factors loss --device cuda
```
