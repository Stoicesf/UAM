# SECDO v2 — Final Theory Freeze

**Status:** FROZEN · Implementation + Paper Production  
**Do not edit** `paper/ac_dsgf_tro/`.

---

## Positioning

SECDO studies evolving feasible regions:
\[
\boxed{
\mathcal{B}_t
\;\rightarrow\;
\hat{\mathcal{B}}_{t+1}
\;\rightarrow\;
\Pi_{\hat{\mathcal{B}}_{t+1}}
\;\rightarrow\;
x_{t+1}
}
\]
Not “prediction + PGD”, but learning-driven anticipation of constraints.

---

## Notation (frozen)

| Symbol | Meaning |
|--------|---------|
| \(c_t\) | scalar capacity / budget teacher |
| \(\mathcal{B}(c)=\{x\ge0:\mathbf{1}^\top x\le c\}\) | feasible set |
| \(\hat c_{t+1}=\mathcal{F}_\phi(h_t)\) | predicted capacity |
| \(\delta_t=\lvert\hat c_{t+1}-c_{t+1}\rvert\) | prediction error (Thm1: \(d_H\le\delta\)) |
| \(\chi_t=\lvert c_{t+1}-c_t\rvert\) | constraint drift (\(=d_H(\mathcal{B}_{t+1},\mathcal{B}_t)\) for scalar budget) |
| \(\mathrm{PI}_t=\delta_t/(\chi_t+\varepsilon)\) | predictability index |
| \(\alpha_t=1/(1+\mathrm{PI}_t^2)\) | anticipation weight |
| \(c^{\mathrm{mix}}_t=\alpha_t\hat c_{t+1}+(1-\alpha_t)c_t\) | mixed budget |
| \(\epsilon_t=\|s_t-\hat s_t\|\) | state prediction error |

Legacy alias: \(\rho_t\equiv\chi_t\) in older Phase-2 code/logs.

---

## Theorem 1 — Constraint Dynamics Stability

For scalar budget \(\sum_i x_i\le c_t\) and prediction \(\hat c_t\),
\[
\boxed{
d_H\bigl(\mathcal{B}(c_t),\mathcal{B}(\hat c_t)\bigr)
\;\le\;
\delta_t
\;=\;
\lvert c_t-\hat c_t\rvert.
}
\]
Ref: `thm1_scalar_budget_stability.md`.

---

## Assumption 3 — Bounded Projection Sensitivity

On compact convex polyhedra, \(\exists C_\Pi<\infty\) s.t.
\[
\bigl\|
\Pi_{\hat{\mathcal{B}}}(y)-\Pi_{\mathcal{B}}(y)
\bigr\|
\;\le\;
C_\Pi\,d_H(\hat{\mathcal{B}},\mathcal{B}).
\]

---

## Lemma 2 — Projection Error Propagation

\[
\|x_{t+1}-x_{t+1}^\star\|
\;\le\;
\|y_t-x_t^\star\|
\;+\;
\|x_t^\star-x_{t+1}^\star\|
\;+\;
C_\Pi\delta_t,
\]
with
\[
\|x_t^\star-x_{t+1}^\star\|
\;\le\;
\frac1\mu\bigl(\omega_t+L_g\chi_t\bigr).
\]
Ref: `anticipatory_projection_lemma.md` (updated language: \(\chi\) replaces \(\rho\)).

---

## Theorem 2 — Dynamic Feasible Variational Regret

Let \(P_T=\sum_t\|x_{t+1}^\star-x_t^\star\|\). Then
\[
\boxed{
\mathrm{Reg}_T
\;\le\;
O\bigl(\sqrt{T(1+P_T)}\bigr)
\;+\;
O\Bigl(\sum_t(\epsilon_t+\delta_t)\Bigr),
}
\]
and
\[
P_T
\;\le\;
\frac1\mu\sum_t(\omega_t+L_g\chi_t).
\]
Ref: `thm2_prediction_aware_gap.md` (regret form = corollary of gap bound).

---

## Theorem 3 — Anticipatory Projection Advantage

Define certificates \(V_A(t)\le\delta_t\), \(V_R(t)\le\chi_t\).  
If \(\mathrm{PI}_t<1\) (i.e. \(\delta_t<\chi_t\)), then
\[
\boxed{
\delta_t<\chi_t
\;\Rightarrow\;
\text{anticipatory certificate strictly tighter than reactive}.
}
\]

---

## Lemma 3 — Surrogate Consistency

Training \(L_c=\|c-\hat c\|^2\) with Lipschitz teacher map implies
\[
\boxed{
\delta_t
\;\le\;
L_g\sqrt{L_c}
}
\]
(up to constant factors / batch averaging). Thus reducing \(L_c\) reduces the theory term.

---

## Corollary 4 — Graceful Recovery

When prediction fails (\(\widehat{\mathrm{PI}}_t>1\)), set
\[
\alpha_t=\frac{1}{1+\mathrm{PI}_t^2}
\;\rightarrow\;0,
\]
so \(c^{\mathrm{mix}}\to c_t\) (reactive). Then
\[
\mathrm{Reg}_{\mathrm{SECDO}}
\;\le\;
\mathrm{Reg}_R
\;+\;
O\Bigl(\sum_t I_t^{\mathrm{fail}}\delta_t\Bigr).
\]

---

## Algorithm 1 (frozen inference)

1. \(z_t=\mathrm{Enc}(s_t),\ h_t=\mathrm{GRU}(z_t,h_{t-1}),\ \hat s_{t+1}=\mathrm{Dec}(h_t)\)  
2. \(\hat c_{t+1}=\mathcal{F}_\phi(h_t)\)  
3. \(y_t=x_t-\eta\nabla F(x_t)\)  
4. \(\mathrm{PI}_t=\hat\delta_t/(\hat\chi_t+\varepsilon),\ \alpha_t=1/(1+\mathrm{PI}_t^2)\)  
5. \(c^{\mathrm{mix}}=\alpha_t\hat c_{t+1}+(1-\alpha_t)c_t,\quad x_{t+1}=\Pi_{\mathcal{B}(c^{\mathrm{mix}})}(y_t)\)  
   with \(\hat c\) stop-grad through \(\Pi\) during training.

---

## Contribution map

1. Learned constraint dynamics \(\mathcal{F}_\phi\) → \(\delta\)  
2. Predictability-conditioned anticipatory projection (\(\mathrm{PI},\alpha\))  
3. Robust dynamic optimization guarantee (Thm1→Lem2→Thm2→Thm3→Cor4)
