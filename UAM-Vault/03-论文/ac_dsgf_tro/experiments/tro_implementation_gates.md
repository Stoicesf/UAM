# T-RO Implementation Gates

**Status:** Gate 0–1 implemented · Gate 2–3 next  
**No training / no reward tuning in this phase.**

Related: [`tro_logging_protocol.md`](tro_logging_protocol.md), [`tro_evaluation_protocol.md`](tro_evaluation_protocol.md)

---

## Gate 0 — Run layout

```text
runs/<run_id>/
  config.json
  topology/     # A_t, degrees, V_B per step
  messages/     # Gate 2 twin M*, M
  actions/      # Gate 2 a*, a
  rewards/
  metrics/
```

Helper: `utils/tro_run.py` → `create_run_dir`, `append_topology_step`.

---

## Gate 1 — Topology projection + \(V_B=0\)  ✅ **PASS** (2026-07-18)

**Question:** Is \(\Pi_{B_t}\) correctly implemented?

**Code:**
- `models/communication/budget_layer.py` — `apply_topk_fixed_k`, `degree_budget_violation`
- `models/ac_dsgf.py` — `fixed_k`, diagnostics `A_t`, `V_B`, `C_t`, `B_t`, `rho_t`
- `algorithms/guided/mappo_guided.py` — `set_fixed_k`, `last_topology_diag`
- `utils/tro_run.py` — `runs/<id>/topology|metrics/...`

**Validate (no training):**
```bash
conda run -n dpg_hrl python scripts/validate_tro_budget_feasibility.py
conda run -n dpg_hrl python scripts/validate_tro_budget_feasibility.py --with-encoder
```

**Result:** synthetic fixed-\(K\) + ratio + untrained encoder dry-run — all `max_V_B == 0` for \(N\in\{8,16,32\}\), \(K\in\{2,4,8\}\).  
Artifact: `experiments/_gate1_last_result.json`, logs under `runs/tro_gate1_enc_*`.

**Forbidden:** changing reward, LR, scorer, or network size to “make Gate 1 pass.”

---

## Gate 2 — Twin evaluation  ✅ **PASS** (2026-07-18)

**Question:** Can \(\varepsilon_G=\|M(G^\star)-M(G_t)\|\) and \(\Delta a=\|a^\star-a\|\) be measured on a **shared** \(s_t\)?

**Code:**
- `models/ac_dsgf.py` — `build_shared_context`, `decode_from_gate`, `twin_forward`
- `tro/twin/` — `TwinEvaluator`, `message_compare`, `action_compare`
- `scripts/validate_tro_twin_evaluation.py`

**Validate:**
```bash
conda run -n dpg_hrl python scripts/validate_tro_twin_evaluation.py
```

**Sanity results:**
| Case | Expectation | Result |
|------|-------------|--------|
| 1 \(G=G^\star\) | \(\varepsilon_G=\Delta a=0\) | PASS |
| 2 \(G=\emptyset\) | \(\varepsilon_G>0\) | PASS |
| 3 \(K\uparrow\) | \(\varepsilon_G\) tends ↓ | PASS (2→4→8) |
| Logs | recover \(\varepsilon_G,\Delta a,M\) | PASS |

**Next:** Gate 2C / formal §6.3 twin figures — still no training for hooks; use frozen checkpoints only when collecting evidence.

---

## Gate 3 / Phase 6.3 — Theory evidence  ✅ smoke collected

**Script:** `scripts/collect_tro_theory_evidence.py`  
**Ckpt:** `results/ac_dsgf/uav16/s1234/checkpoints/final.pt` (frozen)  
**Outputs:** `experiments/evidence_6_3/`, `figures/Fig1–3_*.png`

| Phase | Result (8 ep smoke) |
|-------|---------------------|
| A Lemma 2 | \(D_G\)–\(\varepsilon_G\) Pearson ≈ 0.96; both ↓ as \(K\uparrow\) |
| B Thm 2 mid | \(\varepsilon_G,\Delta a,\Delta J_{\mathrm{act}}\) jointly ↓ |
| C Pareto | Fig.~3 written; enlarge \(n_{\mathrm{ep}}\) for camera-ready |

**No tuning** of encoder / scorer / budget / reward during collection.
