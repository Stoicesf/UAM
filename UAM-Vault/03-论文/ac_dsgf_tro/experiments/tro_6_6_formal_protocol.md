# §6.6 Communication MARL Baselines — Formal Protocol

**Status:** Phase A active · Class C (TarMAC / IC3Net / ATOC) deferred until wrappers exist  
**Purpose:** Answer the reviewer question — *why is learned topology better than existing learned / fixed communication?*  
**No training in Phase A.** Reuse frozen 16-UAV checkpoints only.

---

## Claim hygiene

Forbidden:
> AC-DSGF outperforms all communication learning methods.

Required framing (three steps):
1. **Class A:** communication is useful vs no-comm MAPPO.  
2. **Class B / G2:** under comparable or lower \(C\), dynamic / projected topology beats fixed-support or dense attention on the joint \((J,C)\) plane.  
3. **Class C (later):** topology-as-decision \(G_t=\phi_\theta(s_t)+\Pi_{B_t}\) differs from message learning on a fixed support.

Never compare under silently unequal budgets.

---

## Phase A method freeze (available now)

| Class | Method | Checkpoint root | Role |
|-------|--------|-----------------|------|
| **A** | MAPPO | `results/baseline16_seeds/mappo/s{seed}` | no learned communication |
| **B** | GAT-MAPPO | `results/baseline16_seeds/gat/s{seed}` | fixed radius support + attention |
| **B** | DSGF | `results/baseline16_seeds/dsgf/s{seed}` | dynamic sparse graph *without* budget projection as decision |
| **G2** | Full Attention | `results/baseline16_seeds/transformer/s{seed}` | dense / performance–cost ceiling |
| **Ours** | AC-DSGF | `results/ac_dsgf/uav16/s{seed}` | topology policy + hard \(\Pi_{B_t}\) |

**Seeds:** \(\{1234,2026,3407,42,8888\}\) · **Task:** T1 navigation · **\(N=16\)**

### Phase B (deferred — not blocking Phase A write-up)

TarMAC, IC3Net, ATOC (, MAGIC) under **identical-budget** slices (G1).  
Do not invent numbers. Implement wrappers → then collect.

---

## Metrics (joint plane)

| Symbol | Definition |
|--------|------------|
| \(J\) | episode return (mean over episodes, then seeds) |
| Success | goal reach rate |
| Collision | collision rate |
| \(C\) | mean communication cost — **method-aware** (see below) |
| \(\rho\) | \(C / [N(N-1)]\) when \(C\) is edge count |

**\(C\) sources (locked).**

| Method | \(C\) definition |
|--------|------------------|
| MAPPO | \(0\) (no learned communication) |
| GAT / DSGF | radius-neighborhood edge count |
| Full Attention | dense \(N(N-1)\) |
| AC-DSGF | hard / soft projected edge count \(C_t\) from topology diagnostics |

**Primary figure:** \((C,J)\) or \((C,\mathrm{Success})\) scatter / Pareto.  
**Primary table:** mean±std over 5 seeds.

### AC-DSGF budget slices (identical-budget narrative)

Evaluate frozen AC-DSGF under hard fixed-\(K\in\{2,4,6\}\) plus the training-time soft gate (default).  
Compare Class B methods against the AC-DSGF slice whose \(C\) is closest (disclose \(C\)).

---

## Eval settings

| Item | Spec |
|------|------|
| Episodes / seed | **32** (Phase A); camera-ready may raise to 64–200 |
| `max_steps` | 128 |
| Device | cuda if available |
| Aggregation | mean ± std over seeds |

Modes:
- `summary` — aggregate existing `summary.json` / `paper_metrics` (sanity / draft)  
- `reeval` — unified closed-loop re-evaluation (formal Table II)

---

## Outputs

```text
paper/ac_dsgf_tro/experiments/evidence_6_6_baselines/
├── README.md
├── baseline_comparison.csv
├── evidence_6_6_report.json
└── figures/
paper/ac_dsgf_tro/figures/Fig5_baseline_pareto.png
```

---

## Command

```bash
# Draft from frozen summaries (instant)
E:\ANACONDA\envs\dpg_hrl\python.exe -u scripts/collect_tro_baseline_comparison.py --mode summary

# Formal re-eval
E:\ANACONDA\envs\dpg_hrl\python.exe -u scripts/collect_tro_baseline_comparison.py `
  --mode reeval --episodes 32 --seeds 1234 2026 3407 42 8888 `
  --ac-K 2 4 6 --device cuda
```

---

## Pass criteria (for writing, not hacking)

- All five seeds present per method.  
- Table reports \((J,C)\) jointly.  
- AC-DSGF \(K\)-slices disclose \(C\).  
- No Class C numbers until implementations exist.
