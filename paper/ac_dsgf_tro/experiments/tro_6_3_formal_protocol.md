# §6.3 Formal Evidence Protocol (Frozen)

**Status:** **FROZEN in manuscript** (EN/CN v0.6 §6.3)  
**Do not change mid-campaign.** Smoke results under `evidence_6_3/` are superseded by `evidence_6_3_formal/`.

Manuscript claim form:
> The empirical relationship is consistent with … characterized in Lemma 2 / Lemma 1.  
> Not: “validates Lemma 2.”

---

## Terminology lock

| Forbidden | Required |
|-----------|----------|
| full communication graph | **full-support reference topology before budget projection** |
| \(\Delta J_{\mathrm{act}}\) as “return gap” | \(\Delta A_\gamma=\sum_t\gamma^t\|a_t^\star-a_t\|\) (**discounted action discrepancy**) |
| “experiments prove Theorem 2” | “consistent with Lemma 1 / Lemma 2 trends” |

\[
S_t=\phi_\theta(s_t),\qquad
G_t^\star=\text{pre-projection full-support scores on }A_t,\qquad
G_t=\Pi_{B_t}(S_t).
\]

---

## Figure freeze

| Fig | Axes | Theory | Stats |
|-----|------|--------|-------|
| **1** | \(D_G\to\varepsilon_G\) | Lemma 2 | scatter + linear fit \(\varepsilon_G=\alpha D_G+\beta\), report \(\alpha,R^2\) |
| **2** | \(\varepsilon_G\to\Delta A_\gamma\) (+ \(\Delta a\)) | Lemma 1 | mean±std over seeds; monotone trend |
| **3** | \(B\to J\) / success / \(C\) | Pareto | mean±std over seeds |

---

## Sampling freeze (avoid \(K\)-saturation)

- \(N=16\) (this round; \(N=32\) later if needed)  
- \(K\in\{1,2,4,6\}\) — **not** \(\{8,16\}\) (smoke showed radius saturation)  
- Seeds: frozen ckpts `{1234,2026,3407,42,8888}`  
- Episodes: **32** per seed (camera-ready may raise to 64)  
- No encoder / scorer / budget / reward edits  
- No Theorem 3 / no baselines in this round  

---

## Command

```bash
E:\ANACONDA\envs\dpg_hrl\python.exe scripts/collect_tro_theory_evidence.py `
  --phase ABC --episodes 32 --K 1 2 4 6 `
  --seeds 1234 2026 3407 42 8888 `
  --device cuda
```

Default device is **cuda** (falls back to cpu if unavailable).  
Outputs: `experiments/evidence_6_3_formal/`, `figures/Fig1–3_*.png`
