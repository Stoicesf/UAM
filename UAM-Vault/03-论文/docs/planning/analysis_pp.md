# AC-DSGF++ Phase 6 — Analysis & Acceptance Criteria

**Claim lock:** Performance Preservation + Communication Reduction  
**Not:** Success leadership

Frozen AC-DSGF (Table I, 16UAV × 5 seeds):

| Method | Success | Comm | CEI |
|--------|---------|------|-----|
| GAT | 2.04±0.90% | 38.80 | 0.0005 |
| DSGF | 4.01±1.68% | 39.27 | 0.0010 |
| **AC-DSGF** | **3.95±0.83%** | **0.43** | **0.091** |

---

## Priority order (do not reverse)

1. **Success 保持**（≥3.5% 理想；≈ AC-DSGF 可接受）
2. **corr(U,U*) > 0.5**（证明不是随机关通信）
3. **Comm 下降**（目标 <0.25；最佳 0.1–0.2）

若 Comm↓90% 且 Success 明显掉 → **Case C**：只调 λ 权重，不改结构。

---

## Case triage (after 5 seeds or after seed42 peek)

### Case A — 冲 RA-L / T-RO
```
Success ≈ 4% (≥3.5%)
Comm < 0.2
corr(U,U*) > 0.5
```

### Case B — 稳妥可发表
```
Success ≈ 3.5% (comparable to AC-DSGF)
Comm ↓ ≥50% vs AC-DSGF (e.g. ≤0.22)
corr > 0.5
```

### Case C — 需修损失权重
```
Comm ↓90% (e.g. <0.05) AND Success 明显下降
OR corr < 0.4
```
→ 提高 warm-up / 略降 λ_comm / 略升 λ_u；**禁止改 backbone**

---

## Required artifacts

| File | Content |
|------|---------|
| `paper/tables/table1_pp_main.csv` | Table I++ raw |
| `paper/tables/table1_pp_paper.csv` | Human-readable |
| `paper/figures/fig_utility_alignment.png` | corr vs steps |
| `paper/figures/fig_comm_precision.png` | precision curve |
| `paper/docs/planning/analysis_pp.md` | This file + filled verdict |

---

## Table I++ template

**Task Performance and Communication Efficiency Comparison on 16-UAV Swarm**

| Method | Success ↑ | Comm ↓ | CEI ↑ | corr(U,U*) | Comm Prec. |
|--------|-----------|--------|-------|------------|------------|
| GAT | 2.04±0.90 | 38.80 | 0.0005 | — | — |
| DSGF | 4.01±1.68 | 39.27 | 0.0010 | — | — |
| AC-DSGF | 3.95±0.83 | 0.43 | 0.091 | — | — |
| **AC-DSGF++** | ? | ? | ? | ? | ? |

Story line:
> achieving comparable task performance with more efficient / higher-precision communication.

---

## Commands

```bash
# After seed42 finishes (or any single seed)
python scripts/check_pp_seed_health.py --run results/ac_dsgf_pp/uav16/s42

# After all 5 seeds
python scripts/analyze_ac_dsgf_pp16_table.py
```

---

## Phase 7 order (after Phase 6 Case A/B)

1. **w/o Causal Utility**（最重要）
2. Reward Utility vs Action Utility（理论）
3. Random Utility（最低优先）

## Demo (Phase 8)
Title: *From Communication Quantity to Communication Value*  
三栏：AC-DSGF (quantity) | AC-DSGF++ (utility SEND/DROP) | Oracle

---

## Verdict log (fill after runs)

| Seed | Success | Comm | corr | Prec | Notes |
|------|---------|------|------|------|-------|
| 42 | **4.20%** (64-ep) | train~0.45 / eval~0.71 | **0.51** | 0.0† | Success+corr OK; Comm 未明显低于 v1 → **B-weak** |
| 3407 | | | | | |
| 2026 | | | | | |
| 1234 | | | | | |
| 8888 | | | | | |
| **5-seed** | | | | | Case ? |

† Comm Precision=0：当前 g>0.5 阈值对软门控过严，需改用相对阈值或 top-k 定义（不改算法结构）。

## Paper story (seed42 → pending 5-seed)

Do **not** claim “90% communication reduction” yet.

Current locked line:
> AC-DSGF++ learns causal communication utility and maintains task performance under adaptive communication constraints.

If 5-seed Comm still ≈ AC-DSGF but CUD/corr strong → emphasize **communication quality**, not quantity.

### Metric upgrade
- Replace hard Comm Precision (`g>0.5`) with **CUD** = Σ g·U* / Σ g
- Logged as `logs/cud.csv` from remaining seeds onward


## Auto verdict (2026-07-17)
- Case: **C**
- Success: 3.38%
- Comm: 2.5637
- corr: 0.34311672
- precision: 0.0
