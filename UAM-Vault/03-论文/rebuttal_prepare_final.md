# SECDO-v2.0 Final Rebuttal Pack（10 Questions）

**定位：** predictive constraint-evolution · PI-conditioned opt · failure-safe \(\alpha\)  
**禁止自称：** prediction-enhanced PGD / Pareto 全面最优 / achieves the bound  
**深度稿：** `paper/ac_dsgf_v2/reviewer_attack/R{1,2,3}_*.md`

---

## Q1. Why not RL?

SECDO addresses online optimization with evolving feasible sets, where dynamic-regret / feasibility certificates are primary.
Black-box RL typically lacks comparable constraint-evolution guarantees under nonstationary \(\mathcal{B}_t\).

---

## Q2. Why learn constraints instead of (only) states?

Constraint evolution directly determines admissible decisions.
Theorem 1 translates capacity/state error into Hausdorff set uncertainty; \(L_c\) aligns training with that term.

---

## Q3. Does prediction always help?

**No.** Advantage is characterized by \(\mathrm{PI}_t=\delta_t/\chi_t\): anticipatory certificates tighten when \(\delta<\chi\) (Theorem 3).
Otherwise Corollary 4 reduces \(\alpha_t\).

---

## Q4. Is \(\alpha\) theoretically analyzed?

\(\alpha_t=1/(1+\mathrm{PI}_t^2)\) is an **implementation-level robustness** mechanism enabling graceful degradation.
Core theory characterizes ideal anticipatory vs.\ reactive projection (Remark in Algorithm section); \(\alpha\) continuously interpolates those extremes.
Ablation A3 shows it mainly improves robustness, not nominal regret.

---

## Q5. Why is violation higher than reactive?

SECDO trades a conservative feasibility margin for a lower optimization gap (Fig. 5).
Violation remains bounded; Oracle (future \(c_{t+1}\)) is an informational upper bound, not a deployable baseline.

---

## Q6. Does UAV prove the theory?

**No.** Synthetic experiments validate scaling structure (Fig. 4); UAV demonstrates practical optimality–safety behavior.
We never write “UAV verifies Theorem 2.”

---

## Q7. Did you fit the regret bound after seeing data?

Fig. 4 uses a fitted envelope **only for visualization**.
Theorem 2 is an \(O(\cdot)\) structural upper bound; we claim scaling validation, not numerical identification of universal constants.

---

## Q8. Is SECDO just prediction + PGD?

**No.** (i) Object is constraint/feasible-set evolution; (ii) \(\delta\) enters optimization guarantees; (iii) decisions are predictability-conditioned via \(c^{\mathrm{mix}}\); (iv) Training \(\neq\) online Algorithm 1.

---

## Q9. What if the predictor fails badly?

Corollary 4: \(\mathrm{PI}\uparrow\Rightarrow\alpha\to 0\), SECDO gracefully degrades toward reactive projection.
Crash Fig. 2 and Ablation A3 (\(\times 3\) window rise if \(\alpha\equiv 1\)) support failure containment—not “full recovery.”

---

## Q10. Why so many symbols (PI, \(\alpha\), mix, …)? Aren’t these heuristics?

They are one mechanism chain:
\[
\text{constraint evolution}\to\text{anticipatory/mixed projection}\to\text{adaptive robustness}\to\text{regret analysis}.
\]
PI/\(\alpha\) are the conditioning variables of that chain, not independent add-ons.

---

## Quick pointers

| Topic | Artifact |
|-------|----------|
| Scaling | Fig. 4, Thm. 2 |
| Trade-off | Fig. 5 |
| Crash | Fig. 2, Cor. 4 |
| Ablation | Table ablation / Fig. E |
| Evidence map | `paper/ac_dsgf_v2/tables/evidence_chain.md` |
| Claim audit | `python scripts/final_claim_check.py` |
