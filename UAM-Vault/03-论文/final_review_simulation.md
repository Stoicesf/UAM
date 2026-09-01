# Final Review Simulation（Submission Freeze）

**Version:** SECDO-v2.0-submission  
**配套：** `rebuttal_prepare_final.md` · `ac_dsgf_v2/reviewer_attack/R1–R3`

---

## Reviewer 1（Theory）

**Q:** Your regret bound depends on \(\delta\), but \(\delta\) is unknown online.

**A:** \(\delta_t=d_H(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1})\) is an **analytical** quantity in Theorems 1–2.
Online Algorithm 1 uses proxy estimates \((\hat\delta,\hat\chi)\) (e.g.\ EMA / lagged drift) **only** to set the robustness weight \(\alpha_t\).
The regret certificate is not claimed to be numerically evaluated online without \(\delta\).

---

## Reviewer 2（ML）

**Q:** Is SECDO only a predictor + optimizer combination?

**A:** No.
The contribution links **constraint / feasible-set evolution** to optimization:
set mismatch \(\delta\) enters the regret upper bound; predictability \(\mathrm{PI}=\delta/\chi\) conditions when anticipation tightens feasibility certificates; failure-safe mixing is Corollary 4—not an unrelated forecast head on PGD.

---

## Reviewer 3（Control / Systems）

**Q:** Why is violation sometimes higher than reactive?

**A:** SECDO intentionally avoids overly conservative feasible decisions and trades a small feasibility margin for a lower optimization gap (Fig. 5, optimality–safety trade-off).
Violation remains bounded; crash tests show adaptive \(\alpha\) contains predictor failures.

---

## Extra quick hits

| Attack | One-line defense |
|--------|------------------|
| Fitted Fig. 4 bound | Visualization envelope only; scaling validation |
| UAV verifies theory | No; synthetic = scaling, UAV = system behavior |
| Two projections | We use single \(\Pi_{\mathcal{B}(c^{\mathrm{mix}})}\) |
| Too many mechanisms | One chain: evolution → projection → robustness → regret |
