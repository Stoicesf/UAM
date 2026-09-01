# §6.2 Formal Budget Feasibility Protocol

**Status:** **FROZEN in manuscript** (EN/CN v0.7 §6.2) · gate passed  
**Purpose:** Direct implementation evidence for Theorem 1 — not a performance experiment.  
**No training. No encoder / scorer / reward edits.**

---

## Claim hygiene

Forbidden:
> Experimental results prove Theorem 1.

Required:
> The projection layer consistently satisfies the predefined communication budgets across different swarm sizes and constraints, demonstrating the practical feasibility of the proposed constrained topology decision mechanism.

---

## Title (manuscript)

**6.2 Budget-Constrained Topology Projection Analysis**

Not: “Performance under communication budget” (that conflicts with §6.3).

---

## Factors (frozen)

| Factor | Levels |
|--------|--------|
| \(N\) | \(\{8,16,32,64\}\) |
| Fixed degree \(K\) | \(\{1,2,4,6,8\}\) |
| Ratio \(\rho_B\) | \(\{0.05,0.1,0.2,0.4\}\) |
| Seeds | frozen `uav16` ckpts \(\{1234,2026,3407,42,8888\}\) |
| Episodes / setting | **8** × **50** steps (feasibility sample; \(T\) large enough for \(VR\)) |

**Env note.** For \(N\le 16\): closed-loop VMAS rollouts with frozen checkpoints.  
For \(N\in\{32,64\}\): same frozen scorer + hard \(\Pi_{B_t}\) on geometric position samples (avoids VMAS spawn cost; projection operator unchanged). Enlarge `world_spawning_*` when using env at \(N\ge 32\).

---

## Metrics

| Symbol | Definition | Pass |
|--------|------------|------|
| \(VR\) | \(\frac{1}{T}\sum_t \mathbf{1}(V_B(t)>0)\) | \(VR=0\) |
| \(V_{\max}\) | \(\max_t V_B(t)\) | \(0\) |
| \(E_d\) | \(\lvert\bar d - K\rvert\) (fixed-\(K\); compare also to \(\overline{k_{\mathrm{cap}}}\) when support \(<K\)) | near 0 vs cap |
| \(\rho_t\) | \(\lvert E_t\rvert / [N(N-1)]\) | \(\rho\sim O(1/N)\) under fixed \(K\) |

---

## Gate (must pass before writing Table I)

\[
\forall N,K,\rho_B:\quad V_{\max}=0,\quad VR=0.
\]

If fail: **do not tune experiments**. Debug top-\(k\) ties, directed count, self-loops, budget definition consistency.

---

## Outputs

```text
paper/ac_dsgf_tro/experiments/evidence_6_2_budget/
├── README.md
├── budget_statistics.csv
├── degree_statistics.csv
├── evidence_6_2_report.json
└── figures/   (symlink / copies under paper/.../figures)
```

Optional: `Fig4_communication_scaling.png` — \(\rho\) vs \(N\) (no scalability claim).

---

## Command

```bash
E:\ANACONDA\envs\dpg_hrl\python.exe -u scripts/collect_tro_budget_statistics.py `
  --sizes 8 16 32 64 `
  --K 1 2 4 6 8 `
  --ratios 0.05 0.1 0.2 0.4 `
  --seeds 1234 2026 3407 42 8888 `
  --episodes 8 `
  --max-steps 50 `
  --env-max-n 16 `
  --geom-steps 400 `
  --device cuda
```

Smoke (fast):

```bash
... --sizes 8 16 --K 2 4 --ratios 0.1 0.2 --seeds 1234 --episodes 2 --max-steps 50 --device cuda
```
