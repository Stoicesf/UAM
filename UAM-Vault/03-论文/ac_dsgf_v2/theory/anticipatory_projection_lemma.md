# Lemma 2 — Anticipatory Projection under Dynamic Feasible-set Evolution

**Track:** SECDO  
**Status:** Formal · Phase 1c  
**Depends on:** [`thm1_scalar_budget_stability.md`](thm1_scalar_budget_stability.md) (Assumption 3 / \(d_H\le\delta\))  
**Claim hygiene:** This is a **conditional advantage**, not “anticipatory projection always beats reactive projection.”

---

## Scientific intent

Under bounded feasible-set drift and bounded prediction error, the **future feasibility defect** of anticipatory projection is controlled by \(\delta\), while that of reactive projection is controlled by drift \(\rho\).  
Whenever prediction is more accurate than the environment’s constraint change (\(\delta_{t+1}<\rho_t\)), anticipatory projection has a strictly better **upper bound** on next-step violation distance—and the same comparison lifts to cumulative bounds.

Reviewer-safe reading:
\[
\boxed{
\textbf{Prediction accuracy better than constraint drift}
\;\Rightarrow\;
\textbf{anticipatory upper-bound advantage}
}
\]

---

## Definition 2 (Projection operators)

Let \(F(\cdot)\) be a differentiable surrogate objective and \(\eta_t>0\). Define the unconstrained / surrogate step
\[
y_t
\;:=\;
x_t-\eta_t\nabla F(x_t).
\]

Let \(\mathcal{B}_t=\mathcal{B}(c_t)\) and \(\hat{\mathcal{B}}_{t+1}=\mathcal{B}(\hat c_{t+1})\) be scalar-budget feasible sets (Definition 1 of Theorem 1). Denote by \(\Pi_{\mathcal{S}}\) the Euclidean projection onto a closed convex \(\mathcal{S}\).

**Reactive projection**
\[
x_{t+1}^{R}
\;:=\;
\Pi_{\mathcal{B}_t}(y_t).
\]

**Anticipatory projection**
\[
x_{t+1}^{A}
\;:=\;
\Pi_{\hat{\mathcal{B}}_{t+1}}(y_t).
\]

---

## Assumption 2 (Bounded feasible-set evolution)

The true feasible sets satisfy
\[
d_H\bigl(\mathcal{B}_{t+1},\mathcal{B}_t\bigr)
\;\le\;
\rho_t
\]
for some finite drift sequence \((\rho_t)_{t\ge 0}\).

**UAV scalar capacity specialization.** If \(\mathcal{B}_t=\mathcal{B}(C_t)\) with \(C_t\ge 0\), Theorem 1 yields
\[
\rho_t
\;=\;
\lvert C_{t+1}-C_t\rvert
\]
as a valid choice (taking the tight Thm1 constant \(L_c=1\)).

---

## Assumption 3 (Prediction error bound)

By Theorem 1,
\[
d_H\bigl(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1}\bigr)
\;\le\;
\delta_{t+1}
\;:=\;
\lvert\hat c_{t+1}-c_{t+1}\rvert.
\]

---

## Definition 3 (Future violation distance)

\[
V_{t+1}(x)
\;:=\;
\mathrm{dist}\bigl(x,\mathcal{B}_{t+1}\bigr)
\;=\;
\inf_{z\in\mathcal{B}_{t+1}}\|x-z\|_2.
\]

---

## Lemma 2 (Anticipatory vs reactive future-feasibility bounds)

Under Assumptions 2–3 and Definition 2:

**(i) Anticipatory bound.**
\[
\boxed{
V_{t+1}\bigl(x_{t+1}^{A}\bigr)
\;\le\;
\delta_{t+1}.
}
\]

**(ii) Reactive bound.**
\[
\boxed{
V_{t+1}\bigl(x_{t+1}^{R}\bigr)
\;\le\;
\rho_t.
}
\]

**(iii) Conditional pointwise advantage (on upper bounds).**  
If \(\delta_{t+1}<\rho_t\), then the guaranteed upper bound for anticipatory projection is strictly tighter than that for reactive projection. In particular, one **cannot** conclude \(V_{t+1}(x_{t+1}^{A})<V_{t+1}(x_{t+1}^{R})\) in every realization without further structure; what is guaranteed is
\[
\boxed{
\delta_{t+1}<\rho_t
\;\Longrightarrow\;
\bigl[
\text{UB}\bigl(V_{t+1}(x_{t+1}^{A})\bigr)
=
\delta_{t+1}
\bigr]
\;<\;
\bigl[
\text{UB}\bigl(V_{t+1}(x_{t+1}^{R})\bigr)
=
\rho_t
\bigr].
}
\]

**Stronger realization-wise comparison (optional, when both bounds are saturated as worst-case certificates).**  
If one interprets (i)–(ii) as worst-case certificates used for algorithm design, then whenever \(\delta_{t+1}<\rho_t\) the anticipatory certificate dominates the reactive certificate. Empirical smoke tests compare realized \(V\) under shrinking \(C_t\); theory claims **certificate dominance**, not universal pathwise dominance.

---

## Proof

### Part 1 — Anticipatory bound

By definition of Euclidean projection onto a nonempty closed convex set,
\[
x_{t+1}^{A}
=
\Pi_{\hat{\mathcal{B}}_{t+1}}(y_t)
\;\in\;
\hat{\mathcal{B}}_{t+1}.
\]
Therefore
\[
V_{t+1}\bigl(x_{t+1}^{A}\bigr)
=
\mathrm{dist}\bigl(x_{t+1}^{A},\mathcal{B}_{t+1}\bigr)
\;\le\;
\sup_{u\in\hat{\mathcal{B}}_{t+1}}
\mathrm{dist}\bigl(u,\mathcal{B}_{t+1}\bigr)
\;\le\;
d_H\bigl(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1}\bigr).
\]
Assumption 3 yields \(d_H(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1})\le\delta_{t+1}\), hence
\[
V_{t+1}\bigl(x_{t+1}^{A}\bigr)
\;\le\;
\delta_{t+1}.
\]

### Part 2 — Reactive bound

Likewise,
\[
x_{t+1}^{R}
=
\Pi_{\mathcal{B}_t}(y_t)
\;\in\;
\mathcal{B}_t,
\]
so
\[
V_{t+1}\bigl(x_{t+1}^{R}\bigr)
=
\mathrm{dist}\bigl(x_{t+1}^{R},\mathcal{B}_{t+1}\bigr)
\;\le\;
d_H\bigl(\mathcal{B}_t,\mathcal{B}_{t+1}\bigr)
\;\le\;
\rho_t
\]
by Assumption 2. □

---

## Corollary 2.1 (Cumulative future-feasibility certificates)

Summing Lemma 2 over \(t=0,\ldots,T-1\),
\[
\boxed{
\sum_{t=0}^{T-1}
V_{t+1}\bigl(x_{t+1}^{A}\bigr)
\;\le\;
\sum_{t=0}^{T-1}
\delta_{t+1},
}
\qquad
\boxed{
\sum_{t=0}^{T-1}
V_{t+1}\bigl(x_{t+1}^{R}\bigr)
\;\le\;
\sum_{t=0}^{T-1}
\rho_t.
}
\]

Consequently, the gap between cumulative **certificates** satisfies
\[
\sum_{t=0}^{T-1}
\Bigl(
\mathrm{UB}\bigl(V_{t+1}(x_{t+1}^{A})\bigr)
-
\mathrm{UB}\bigl(V_{t+1}(x_{t+1}^{R})\bigr)
\Bigr)
\;\le\;
\sum_{t=0}^{T-1}
(\delta_{t+1}-\rho_t).
\]
Whenever \(\sum_t(\delta_{t+1}-\rho_t)<0\) (prediction more accurate than drift on average), the anticipatory cumulative certificate is strictly better.

---

## Corollary 2.2 (SECDO operating regime)

SECDO’s anticipatory mechanism is justified in the regime
\[
\boxed{
\delta_{t+1}
\;<\;
\rho_t
\quad\text{(or on average }\mathbb{E}[\delta_{t+1}-\rho_t]<0\text{)}.
}
\]
If prediction is poor (\(\delta\gg\rho\)), the lemma **does not** promise improvement over reacting to \(\mathcal{B}_t\); this matches the top-venue requirement that advantage be **conditional** on foresight quality.

---

## Discussion — what this lemma contributes

| Contribution | Non-claim |
|--------------|-----------|
| Separates anticipatory \(\Pi_{\hat{\mathcal{B}}_{t+1}}\) from reactive \(\Pi_{\mathcal{B}_t}\) with explicit \(\delta\) vs \(\rho\) | “Always lower realized violation” |
| Feeds Theorem 3 / narrative: learn \(\mathcal{F}_\phi\) to drive \(\delta\) below drift \(\rho\) | Optimality of \(y_t\) or of \(F\) |
| Uses only convex closed projections + Hausdorff geometry (Thm1) | World-model accuracy rates |

---

## Interface to the three-layer theory chain

\[
\boxed{
\begin{aligned}
\textbf{Thm1:}
&\quad
|\hat c-c|
\;\rightarrow\;
d_H(\hat{\mathcal{B}},\mathcal{B})=\delta
\\
\textbf{Lemma2:}
&\quad
V^{A}\le\delta,\;V^{R}\le\rho;
\quad
\delta<\rho
\;\Rightarrow\;
\text{anticipatory certificate advantage}
\\
\textbf{Thm2:}
&\quad
\epsilon+\delta
\;\rightarrow\;
\text{prediction-aware optimization gap}
\end{aligned}
}
\]

**Next:** Theorem 2 main proof (prediction-aware optimization gap), now that \(\delta\) and anticipatory certificates are rigorously available.
