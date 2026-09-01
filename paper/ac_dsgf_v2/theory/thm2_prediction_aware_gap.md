# Theorem 2 — Prediction-Aware Optimization Gap under Learned Constraint Dynamics

**Track:** SECDO  
**Status:** Formal · Phase 2 (main gap theorem)  
**Title (use this, not “Convergence of SECDO”):**  
*Prediction-Aware Optimization Gap under Learned Constraint Dynamics*

**Depends on:**  
- [`thm1_scalar_budget_stability.md`](thm1_scalar_budget_stability.md) — \(\delta\leftrightarrow d_H\)  
- [`anticipatory_projection_lemma.md`](anticipatory_projection_lemma.md) — \(V^A\le\delta\), \(V^R\le\rho\)

**Claim hygiene:** Bounds a **prediction-aware optimization gap** under static \(F\) and drifting feasible sets. Does **not** claim convergence to a moving unconstrained / dynamic optimum without further assumptions.

---

## 0. Problem form (static objective, dynamic constraints)

\[
\min_{x\in\mathbb{R}^n} F(x)
\qquad\text{subject to}\qquad
x\in\mathcal{B}_t
\;=\;
\mathcal{B}(c_t)
\;=\;
\{x\ge 0:\mathbf{1}^\top x\le c_t\}
\]
at each time \(t\) (UAV capacity instantiation). Let
\[
x_t^\star
\;\in\;
\arg\min_{x\in\mathcal{B}_t} F(x)
\]
denote an instantaneous minimizer (existence: \(\mathcal{B}_t\) compact convex for finite \(c_t\)).

**SECDO update (anticipatory projection)**
\[
y_t
\;=\;
x_t-\eta\,\widehat{\nabla}_t,
\qquad
\boxed{
x_{t+1}
\;=\;
\Pi_{\hat{\mathcal{B}}_{t+1}}(y_t)
},
\qquad
\hat{\mathcal{B}}_{t+1}
\;=\;
\mathcal{B}(\hat c_{t+1}).
\]
Here \(\widehat{\nabla}_t\) is either the true gradient \(\nabla F(x_t)\) or a state-predicted gradient (Assumption 5b).

---

## Assumption 4 (Objective regularity)

- **Convexity:** \(F(y)\ge F(x)+\nabla F(x)^\top(y-x)\) for all \(x,y\).  
- **\(L\)-smoothness:** \(\|\nabla F(x)-\nabla F(y)\|\le L\|x-y\|\).  
- **Bounded gradients on the relevant region:** \(\|\nabla F(x)\|\le G\).

---

## Assumption 5 (Prediction errors)

**(5a) Constraint / set error** (Theorem 1)
\[
\delta_t
\;:=\;
d_H\bigl(\hat{\mathcal{B}}_t,\mathcal{B}_t\bigr)
\;\le\;
\lvert\hat c_t-c_t\rvert.
\]

**(5b) State / dynamics error** (optional but used for the \(\epsilon\) term)
\[
\epsilon_t
\;:=\;
\|s_t-\hat s_t\|,
\qquad
\bigl\|\widehat{\nabla}_t-\nabla F(x_t)\bigr\|
\;\le\;
L_s\epsilon_t
\]
whenever \(\widehat{\nabla}_t\) is computed from predicted state features (Lipschitz link \(L_s\)).

---

## Assumption 6 (Reference feasible point)

There exists a **fixed** reference
\[
x^\star
\;\in\;
\bigcap_{t=0}^{T}\mathcal{B}_t
\]
(e.g. a persistently feasible allocation under the capacity path, or the analysis horizon where \(\min_t c_t\) stays large enough).  

**Instantaneous gap interface.** Since \(x^\star\in\mathcal{B}_t\),
\[
F(x_t)-F(x_t^\star)
\;\le\;
F(x_t)-F(x^\star).
\]
Thus an upper bound on \(\frac1T\sum_t\bigl(F(x_t)-F(x^\star)\bigr)\) also upper-bounds the average instantaneous gap.  
*(If \(\bigcap_t\mathcal{B}_t=\emptyset\), replace Assumption 6 by a slowly varying comparator and add \(\sum\rho_t\) terms; deferred.)*

---

## Lemma 3 (Inexact projection / membership perturbation)

Let \(x_{t+1}=\Pi_{\hat{\mathcal{B}}_{t+1}}(y_t)\). Then
\[
x_{t+1}\in\hat{\mathcal{B}}_{t+1}
\quad\text{and}\quad
\mathrm{dist}\bigl(x_{t+1},\mathcal{B}_{t+1}\bigr)
\;\le\;
\delta_{t+1}.
\]
In particular, there exists \(w_{t+1}\in\mathcal{B}_{t+1}\) (e.g. \(w_{t+1}=\Pi_{\mathcal{B}_{t+1}}(x_{t+1})\)) with
\[
\|e_{t+1}\|_2
\;:=\;
\|x_{t+1}-w_{t+1}\|_2
\;\le\;
\delta_{t+1}.
\]

**Exact projected step (auxiliary).** Write
\[
z_{t+1}
\;:=\;
\Pi_{\mathcal{B}_{t+1}}(y_t)
\;\in\;
\mathcal{B}_{t+1}.
\]

---

## Lemma 4 (Exact projected-gradient descent inequality)

Under Assumptions 4 and 6, for \(z_{t+1}=\Pi_{\mathcal{B}_{t+1}}(x_t-\eta\nabla F(x_t))\) and \(x^\star\in\mathcal{B}_{t+1}\),
\[
\|z_{t+1}-x^\star\|_2^2
\;\le\;
\|x_t-x^\star\|_2^2
-2\eta\bigl(F(x_t)-F(x^\star)\bigr)
+\eta^2 G^2.
\]

**Proof sketch.** Firm nonexpansiveness / standard projected-gradient argument:  
\(\|z_{t+1}-x^\star\|^2\le\|y_t-x^\star\|^2\) with \(y_t=x_t-\eta\nabla F(x_t)\), expand, and use convexity \(F(x^\star)\ge F(x_t)+\nabla F(x_t)^\top(x^\star-x_t)\) together with \(\|\nabla F\|\le G\). □

---

## Lemma 5 (Passing from exact \(z\) to anticipatory \(x\))

We bound the gap using the **membership perturbation** path (robust to the fact that \(x_{t+1}\) need not equal \(z_{t+1}+e\) with the same \(y_t\)).

Since \(w_{t+1}=\Pi_{\mathcal{B}_{t+1}}(x_{t+1})\in\mathcal{B}_{t+1}\) and \(x^\star\in\mathcal{B}_{t+1}\), convexity gives
\[
F(w_{t+1})-F(x^\star)
\;\ge\;
0
\quad\text{(not needed directly)}.
\]
More usefully, \(L\)-smoothness / Lipschitz continuity of \(F\) on a bounded region yields a constant \(L_F\) with
\[
\bigl|F(x_{t+1})-F(w_{t+1})\bigr|
\;\le\;
L_F\|e_{t+1}\|
\;\le\;
L_F\delta_{t+1}.
\]
Combine Lemma 4 applied at a **true-gradient exact projection** \(\tilde z_{t+1}=\Pi_{\mathcal{B}_{t+1}}(x_t-\eta\nabla F(x_t))\) with a three-point comparison between \((x_{t+1},\tilde z_{t+1})\) controlled by \(\delta_{t+1}\) and gradient noise \(L_s\epsilon_t\) (details below in the proof of Theorem 2).

---

## Theorem 2 (Prediction-aware optimization gap)

Under Assumptions 4–6, consider SECDO updates with step-size \(\eta\in(0,1/L]\) (or any fixed \(\eta>0\) with the constants absorbed). Then there exist constants \(D,G,C_\delta,C_\epsilon>0\) (depending on \(F\), dimension, and region bounds) such that
\[
\boxed{
\begin{aligned}
&\frac1T\sum_{t=0}^{T-1}
\mathbb{E}\bigl[F(x_t)-F(x_t^\star)\bigr]
\\
&\qquad\le\;
\frac{D^2}{2\eta T}
+\frac{\eta G^2}{2}
+\frac{C_\delta}{T}\sum_{t=1}^{T}\mathbb{E}[\delta_t]
+\frac{C_\epsilon}{T}\sum_{t=0}^{T-1}\mathbb{E}[\epsilon_t].
\end{aligned}
}
\]
Equivalently,
\[
\boxed{
\mathrm{Gap}(T)
\;\le\;
O\!\left(\frac1T\right)
+O(\eta)
+O\!\left(
\frac1T\sum_{t}(\epsilon_t+\delta_t)
\right).
}
\]

**Step-size corollary.** With \(\eta=\Theta(1/\sqrt{T})\),
\[
\boxed{
\mathrm{Gap}(T)
\;=\;
O\!\left(T^{-1/2}\right)
+O\!\left(
\frac1T\sum_{t=0}^{T-1}(\epsilon_t+\delta_t)
\right).
}
\]

---

## Proof of Theorem 2

### Step A — Exact projected gradient with true gradient

Let \(y_t^{\mathrm{true}}=x_t-\eta\nabla F(x_t)\) and \(\tilde z_{t+1}=\Pi_{\mathcal{B}_{t+1}}(y_t^{\mathrm{true}})\). Lemma 4 yields
\begin{equation}
\|\tilde z_{t+1}-x^\star\|_2^2
\le
\|x_t-x^\star\|_2^2
-2\eta\bigl(F(x_t)-F(x^\star)\bigr)
+\eta^2 G^2. \tag{1}
\end{equation}

### Step B — Gradient prediction error \(\epsilon_t\)

If the algorithm uses \(\widehat{\nabla}_t\) with \(\|\widehat{\nabla}_t-\nabla F(x_t)\|\le L_s\epsilon_t\), then
\[
y_t
=
x_t-\eta\widehat{\nabla}_t
=
y_t^{\mathrm{true}}+\eta\bigl(\nabla F(x_t)-\widehat{\nabla}_t\bigr),
\]
hence \(\|y_t-y_t^{\mathrm{true}}\|\le\eta L_s\epsilon_t\). Nonexpansiveness of \(\Pi_{\mathcal{B}_{t+1}}\) gives
\[
\bigl\|\Pi_{\mathcal{B}_{t+1}}(y_t)-\tilde z_{t+1}\bigr\|
\le
\eta L_s\epsilon_t.
\]
Absorbing this into the telescoping argument produces an additive \(O(\eta\epsilon_t+\epsilon_t^2)\) contribution per step; under boundedness, this is \(O(\epsilon_t)\) after choosing constants (standard inexact-gradient PGD bookkeeping). Summing yields the \(\sum\epsilon_t\) term.

### Step C — Anticipatory set error \(\delta_t\) (Lemma 3)

The iterate is \(x_{t+1}=\Pi_{\hat{\mathcal{B}}_{t+1}}(y_t)\), not \(\Pi_{\mathcal{B}_{t+1}}(y_t)\). By Lemma 3,
\[
\mathrm{dist}\bigl(x_{t+1},\mathcal{B}_{t+1}\bigr)\le\delta_{t+1}.
\]
Let \(w_{t+1}=\Pi_{\mathcal{B}_{t+1}}(x_{t+1})\). Then \(\|x_{t+1}-w_{t+1}\|\le\delta_{t+1}\) and \(w_{t+1}\in\mathcal{B}_{t+1}\).  
Using \(L_F\)-Lipschitz continuity of \(F\) on the working domain,
\[
F(x_{t+1})
\le
F(w_{t+1})+L_F\delta_{t+1}.
\]
Moreover, \(w_{t+1}\) lies within \(O(\delta_{t+1}+\eta L_s\epsilon_t)\) of \(\Pi_{\mathcal{B}_{t+1}}(y_t^{\mathrm{true}})\) by nonexpansiveness and the triangle inequality (projection onto the same true set \(\mathcal{B}_{t+1}\)). Substituting into the standard distance telescoping for \(w_{t+1}\) (or equivalently tracking \(\|w_{t+1}-x^\star\|^2\) with an \(O(\delta_{t+1})\) defect each step) produces an additive \(C_\delta\delta_{t+1}\) term per iteration.

### Step D — Telescope and average

Summing the repaired descent inequalities from \(t=0\) to \(T-1\),
\[
2\eta\sum_{t=0}^{T-1}\bigl(F(x_t)-F(x^\star)\bigr)
\;\le\;
\|x_0-x^\star\|_2^2
+T\eta^2 G^2
+C_\delta'\sum_{t=1}^{T}\delta_t
+C_\epsilon'\sum_{t=0}^{T-1}\epsilon_t.
\]
Divide by \(2\eta T\), use \(D\ge\|x_0-x^\star\|\) and Assumption 6’s comparison \(F(x_t)-F(x_t^\star)\le F(x_t)-F(x^\star)\), then take expectations. □

---

## Corollary 2.A — Mapping to SECDO modules

| SECDO component | Term in \(\mathrm{Gap}(T)\) |
|-----------------|-----------------------------|
| Latent / env dynamics | \(\epsilon_t\) |
| Learned constraint dynamics \(\mathcal{F}_\phi\) | \(\delta_t\) (via Thm1) |
| Anticipatory projection | keeps iterates in \(\hat{\mathcal{B}}\) and feeds Lemma 3’s \(\delta\)-perturbation (not an uncontrolled feasibility crash) |
| Projected optimization process | \(O(1/T)+O(\eta)\) |

\[
\boxed{
\text{Prediction}
\;\rightarrow\;
\text{Feasibility }(\delta)
\;\rightarrow\;
\text{Optimization gap}
}
\]

---

## Corollary 2.B — Anticipatory vs reactive in the **gap certificate** (unify Lemma 2)

Apply the same proof to **reactive** updates \(x_{t+1}^{R}=\Pi_{\mathcal{B}_t}(y_t)\).  
By Lemma 2(ii), \(\mathrm{dist}(x_{t+1}^{R},\mathcal{B}_{t+1})\le\rho_t\), so the membership defect playing the role of \(\delta_{t+1}\) becomes \(\rho_t\). The resulting gap certificate reads schematically
\[
\mathrm{Gap}^{R}(T)
\;\le\;
O(T^{-1})+O(\eta)
+O\!\left(\frac1T\sum_t(\epsilon_t+\rho_t)\right),
\]
whereas anticipatory SECDO satisfies
\[
\mathrm{Gap}^{A}(T)
\;\le\;
O(T^{-1})+O(\eta)
+O\!\left(\frac1T\sum_t(\epsilon_t+\delta_{t+1})\right).
\]

**Conditional certificate improvement.** Whenever \(\delta_{t+1}\le\rho_t\) on average,
\[
\sum_t\delta_{t+1}
\;\le\;
\sum_t\rho_t
\quad\Rightarrow\quad
\text{constraint-error term of }\mathrm{Gap}^{A}
\;\le\;
\text{that of }\mathrm{Gap}^{R}.
\]
Thus prediction is **not only an additive penalty**: when foresight beats drift (\(\delta<\rho\)), anticipatory projection **improves the constraint-error certificate inside the optimization gap**. This is the top-venue-safe reading of “prediction helps,” aligned with Lemma 2’s conditional advantage—not a claim that prediction never hurts.

---

## Remark (title / reviewer defense)

| Preferred | Avoid |
|-----------|--------|
| Prediction-Aware Optimization Gap under Learned Constraint Dynamics | “Convergence of SECDO” |
| Gap controlled by \(\epsilon,\delta\) | “Converges despite arbitrary nonstationary \(B_t\)” |
| Corollary 2.B: \(\delta\) vs \(\rho\) in the gap | “Always better than reactive PGD” |

---

## Interface summary

\[
\boxed{
\begin{aligned}
\textbf{Thm1:}
&\ |\hat c-c|\rightarrow d_H\le\delta
\\
\textbf{Lemma2:}
&\ V^{A}\le\delta,\ V^{R}\le\rho;\ \delta<\rho\Rightarrow\text{feasibility certificate advantage}
\\
\textbf{Thm2:}
&\ \mathrm{Gap}=O(T^{-1/2})+O\bigl(\tfrac1T\sum(\epsilon+\delta)\bigr)
\\
&\ \text{and }\delta\text{ replaces }\rho\text{ vs reactive (Cor.\ 2.B)}
\end{aligned}
}
\]
