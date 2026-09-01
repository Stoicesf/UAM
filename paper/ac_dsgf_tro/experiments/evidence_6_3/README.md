# §6.3 Theory Evidence (frozen checkpoint)

**Status:** Phase A/B/C collected · no architecture / reward / scorer changes  
**Checkpoint:** `results/ac_dsgf/uav16/s1234/checkpoints/final.pt`  
**Script:** `scripts/collect_tro_theory_evidence.py`  
**Figures:** `paper/ac_dsgf_tro/figures/Fig{1,2,3}_*.png`

## Claim hygiene

> Empirical trends are **consistent with** Lemma 2 / Theorem 2; they do **not** prove the theorems.  
> Fig.~2 primary axis is shared-state **action-gap** \(\Delta J_{\mathrm{act}}=\sum_t\gamma^t\|a_t^\star-a_t\|\) (Theorem 2 proof intermediate). Navigation task reward is largely state-based, so twin task-\(\Delta J\) is not used as the primary panel.

## Phase A — Lemma 2 (\(D_G\to\varepsilon_G\))

| \(K\) | mean \(D_G\) | mean \(\varepsilon_G\) |
|------|-------------|----------------------|
| 1 | 5.45 | 2.82 |
| 2 | 3.44 | 1.07 |
| 4 | 1.27 | 0.19 |
| 8 | 0.00 | 0.00 |
| 16 | 0.00 | 0.00 |

Pearson (K-means): **0.96** for \(D_G\)–\(\varepsilon_G\).

## Phase B — Thm 2 intermediate (\(\varepsilon_G\to\Delta a,\Delta J_{\mathrm{act}}\))

| \(K\) | \(\varepsilon_G\) | \(\Delta a\) | \(\Delta J_{\mathrm{act}}\) |
|------|------------------|-------------|---------------------------|
| 1 | 2.82 | 0.143 | 5.47 |
| 2 | 1.07 | 0.077 | 2.90 |
| 4 | 0.19 | 0.009 | 0.31 |
| 8–16 | 0 | 0 | 0 |

Monotone: larger budget → smaller information & action gaps (E1 twin).

## Phase C — Budget–reward Pareto (E0, 8 ep smoke)

From first ABC run (`tro_evidence_6_3_20260718_214912`): full vs fixed-\(K\) / ratio. See `phaseC_summary.csv` and Fig.~3. **Smoke-scale** (8 episodes) — enlarge before camera-ready.

## Twin reference fix

\(G^\star\) uses **pre-projection scores on full support** (not binary mask alone), so when Top-\(K\) retains all edges, \(\varepsilon_G\to 0\) as required for sanity.

## Re-run

```bash
E:\ANACONDA\envs\dpg_hrl\python.exe scripts/collect_tro_theory_evidence.py --phase ABC --episodes 32 --max-steps 64 --seed 1234
```
