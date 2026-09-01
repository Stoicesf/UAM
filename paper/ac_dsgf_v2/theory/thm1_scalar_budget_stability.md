# Theorem 1 — Scalar Budget Feasible-set Stability

**Track:** SECDO (Self-Evolving Constrained Distributed Optimization)  
**Status:** Formal · Phase 1b complete · ready to cite from Thm2  
**Code twin:** `models/secdo/theory/feasible_set_geometry.py` · teacher `uav_capacity_teacher.py`  
**Does not claim:** optimality of \(x\); closed-loop control; prediction learning rates.

---

## Role in the SECDO chain

\[
\boxed{
\text{ConstraintHead error }|\hat c-c|
\;\Longrightarrow\;
\text{feasible-set error }d_H(\hat{\mathcal{B}},\mathcal{B})
}
\]

This theorem supplies the **geometric meaning** of the \(\delta_t\) term in Theorem 2 (prediction-aware optimization gap). It is intentionally elementary: the scientific value is the **bridge**, not novelty of Hausdorff geometry alone.

---

## Definition 1 (Scalar-budget feasible set)

For a resource budget \(c\ge 0\) and allocation dimension \(n\in\mathbb{N}\), define
\[
\mathcal{B}(c)
\;:=\;
\Bigl\{
x\in\mathbb{R}^{n}_{\ge 0}
\;:\;
\mathbf{1}^\top x\le c
\Bigr\}.
\]

At time \(t\), let \(c_t\ge 0\) be the **true** capacity (UAV Shannon teacher) and \(\hat c_t\ge 0\) a **predicted** capacity. Write
\[
\mathcal{B}_t
\;:=\;
\mathcal{B}(c_t),
\qquad
\hat{\mathcal{B}}_t
\;:=\;
\mathcal{B}(\hat c_t).
\]

*(Instantiation: \(c_t=C_t=W\log_2(1+\overline{\mathrm{SINR}}_t)\); see `phase1b_uav_c_teacher.md`.)*

---

## Assumption 1 (Budget prediction error)

Define
\[
\delta_t
\;:=\;
\lvert c_t-\hat c_t\rvert.
\]
Assume \(\delta_t<\infty\) (always true for finite real capacities). No statistical assumption is required for Theorem 1.

---

## Theorem 1 (Scalar budget feasible-set stability)

Let \(\mathcal{B}_t=\mathcal{B}(c_t)\) and \(\hat{\mathcal{B}}_t=\mathcal{B}(\hat c_t)\) be as in Definition 1, and let \(d_H\) denote the Euclidean Hausdorff distance on \(\mathbb{R}^n\):
\[
d_H(A,B)
\;:=\;
\max
\Bigl\{
\sup_{a\in A}\mathrm{dist}(a,B),\;
\sup_{b\in B}\mathrm{dist}(b,A)
\Bigr\},
\qquad
\mathrm{dist}(z,S)=\inf_{s\in S}\|z-s\|_2.
\]
Then
\[
\boxed{
d_H\bigl(\mathcal{B}_t,\hat{\mathcal{B}}_t\bigr)
\;\le\;
\lvert c_t-\hat c_t\rvert
\;=\;
\delta_t.
}
\]

---

## Proof

Without loss of generality, assume \(c_t\ge\hat c_t\) (the case \(\hat c_t\ge c_t\) is symmetric by swapping the two sets). Then \(\hat{\mathcal{B}}_t\subseteq\mathcal{B}_t\), which immediately yields
\[
\sup_{x\in\hat{\mathcal{B}}_t}\mathrm{dist}\bigl(x,\mathcal{B}_t\bigr)
\;=\;
0.
\]

It remains to bound \(\sup_{x\in\mathcal{B}_t}\mathrm{dist}(x,\hat{\mathcal{B}}_t)\). Fix arbitrary \(x\in\mathcal{B}_t\), so \(x\ge 0\) and \(\mathbf{1}^\top x\le c_t\).

**Case A.** If \(\mathbf{1}^\top x\le\hat c_t\), then \(x\in\hat{\mathcal{B}}_t\), hence \(\mathrm{dist}(x,\hat{\mathcal{B}}_t)=0\).

**Case B.** If \(\mathbf{1}^\top x>\hat c_t\), necessarily \(\hat c_t<\mathbf{1}^\top x\le c_t\). Define the radial scaling
\[
\tilde x
\;:=\;
\frac{\hat c_t}{\mathbf{1}^\top x}\,x.
\]
Then \(\tilde x\ge 0\) and \(\mathbf{1}^\top\tilde x=\hat c_t\), so \(\tilde x\in\hat{\mathcal{B}}_t\). Moreover,
\[
\|x-\tilde x\|_2
=
\Bigl(1-\frac{\hat c_t}{\mathbf{1}^\top x}\Bigr)\|x\|_2.
\]
For \(x\ge 0\), \(\|x\|_2\le\|x\|_1=\mathbf{1}^\top x\), therefore
\[
\|x-\tilde x\|_2
\le
\Bigl(1-\frac{\hat c_t}{\mathbf{1}^\top x}\Bigr)\mathbf{1}^\top x
=
\mathbf{1}^\top x-\hat c_t
\le
c_t-\hat c_t.
\]
Hence \(\mathrm{dist}(x,\hat{\mathcal{B}}_t)\le c_t-\hat c_t\).

Combining Cases A–B,
\[
\sup_{x\in\mathcal{B}_t}\mathrm{dist}\bigl(x,\hat{\mathcal{B}}_t\bigr)
\le
c_t-\hat c_t
=
\lvert c_t-\hat c_t\rvert.
\]
Together with the inclusion direction,
\[
d_H\bigl(\mathcal{B}_t,\hat{\mathcal{B}}_t\bigr)
\le
\delta_t.
\]

The symmetric argument with \(c_t\leftrightarrow\hat c_t\) covers \(\hat c_t\ge c_t\). □

---

## Corollary 1.1 (Learning error ⇒ constraint error)

If a predictor outputs \(\hat c_t\) with absolute error \(\delta_t=\lvert\hat c_t-c_t\rvert\), then the induced feasible sets satisfy
\[
\boxed{
d_H\bigl(\mathcal{B}(c_t),\mathcal{B}(\hat c_t)\bigr)
\le
\delta_t.
}
\]
In particular, for SECDO’s ConstraintHead / \(\mathcal{F}_\phi\),
\[
\text{capacity MSE / MAE}
\;\Longrightarrow\;
\text{Hausdorff feasible-set mismatch}.
\]

---

## Corollary 1.2 (Projection lands in the predicted set)

Let \(\Pi_{\hat{\mathcal{B}}_t}\) denote Euclidean projection onto \(\hat{\mathcal{B}}_t\). For any \(y_t\in\mathbb{R}^n\),
\[
x_{t+1}
=
\Pi_{\hat{\mathcal{B}}_t}(y_t)
\;\in\;
\hat{\mathcal{B}}_t
=
\mathcal{B}(\hat c_t).
\]
Combined with Theorem 1,
\[
\mathrm{dist}\bigl(x_{t+1},\mathcal{B}_t\bigr)
\le
d_H\bigl(\hat{\mathcal{B}}_t,\mathcal{B}_t\bigr)
\le
\delta_t.
\]
Thus anticipatory projection is **exactly feasible for \(\hat{\mathcal{B}}_t\)** and **approximately feasible for \(\mathcal{B}_t\)** with explicit defect \(\delta_t\).

---

## Remark (what this is / is not)

| Is | Is not |
|----|--------|
| A sharp, code-aligned geometric lemma for scalar UAV capacity | A convergence theorem for \(F(x_t)\) |
| The rigorous gateway for Thm2’s \(O(\sum\delta_t)\) term | A claim that \(\hat c\) is learned consistently |
| Compatible with sum-budget smoke projection | Automatically true for arbitrary non-monotone constraint families |

For non-scalar / non-nested feasible families, replace Theorem 1 by a Lipschitz set-valued assumption \(d_H(\mathcal{B}(c),\mathcal{B}(\hat c))\le L_c\lvert c-\hat c\rvert\); the SECDO v1 instantiation uses \(L_c=1\).

---

## Pointers

| Item | Path |
|------|------|
| Informal bridge note | [`delta_to_hausdorff.md`](delta_to_hausdorff.md) |
| UAV \(c_t\) teacher | [`../phase1b_uav_c_teacher.md`](../phase1b_uav_c_teacher.md) |
| Thm2 target | [`../theory_targets.md`](../theory_targets.md) |
| **Next** | Anticipatory Projection Lemma (Phase 1c) |
