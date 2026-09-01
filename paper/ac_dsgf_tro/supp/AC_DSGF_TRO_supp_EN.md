# Supplementary Material — AC-DSGF-TRO (v0.11)

**Main manuscript:** [`../AC_DSGF_TRO_EN.md`](../AC_DSGF_TRO_EN.md)  
**Status:** Submission package · **no new scientific claims**  
**Class C:** deferred (not required)

This supplement expands proofs, algorithm / training details, experimental settings, and reproducibility notes already frozen in the main text and `theory/` / `experiments/` folders.

---

## Contents

| Section | Content |
|---------|---------|
| [S1](#s1-additional-theoretical-details) | Proofs of Thm.~1, Lem.~2, Thm.~2; Prop. complexity |
| [S2](#s2-algorithm-details) | Topology projection; joint training of \((\phi_\theta,\pi_\psi)\) |
| [S3](#s3-experimental-details) | Environment, hyperparameters, communication accounting |
| [S4](#s4-additional-results) | Pointers to frozen evidence (no new experiment directions) |
| [S5](#s5-reproducibility-checklist) | Checklist |

Canonical proof sources (authoritative):  
[`../theory/theorem_budget.md`](../theory/theorem_budget.md) ·  
[`../theory/lemma_information_discrepancy.md`](../theory/lemma_information_discrepancy.md) ·  
[`../theory/theorem_return_bound.md`](../theory/theorem_return_bound.md) ·  
[`../theory/proposition_complexity.md`](../theory/proposition_complexity.md)

---

## S1. Additional Theoretical Details

### S1.1 Proof of Theorem 1

**Statement (feasibility of budget-constrained dynamic topology projection).**  
For \(G_t=\Pi_{B_t}(S_t)\): (1) \(C(G_t)\le B_t\) (or \(\lvert E_t(i)\rvert\le K\)); (2) utility-maximizing projection under learned scores—**not** task-return optimality.

**Proof sketch.**  
(1) Holds by construction of the feasible set of \(\Pi_{B_t}\).  
(2) Holds by definition of the constrained argmax. Under separable Top-\(K\), each row keeps the \(K\) largest admissible scores.

**Emphasis for reviewers.** Soft training with \(\lambda c(G)\) is a Lagrangian *relaxation*. Hard feasibility is an **evaluation / rollout operator** property: Theorem 1 applies whenever \(\Pi_{B_t}\) is executed—not because SGD alone enforces \(B_t\).

Full write-up: [`proofs/thm1_budget.md`](proofs/thm1_budget.md).

---

### S1.2 Proof of Lemma 2

**Object.** Topology-induced information discrepancy
\[
\varepsilon_G(t)=\bigl\|M(G_t^\star)-M(G_t)\bigr\|,
\qquad
d_G(G_t,G_t^\star)=\|A_t-A_t^\star\|_F,
\]
with \(G_t^\star\) = **full-support reference topology before projection** (not an optimum).

**Lemma 2.** Under message-map Lipschitzness (Assumption M),
\[
\varepsilon_G(t)\le L_M\,d_G(G_t,G_t^\star).
\]

**Proof.** Apply Assumption M to \((G_t^\star,G_t)\). Finiteness follows for finite \(N\).

Full write-up: [`proofs/lemma2_information.md`](proofs/lemma2_information.md).

---

### S1.3 Proof of Theorem 2

**Official title.** Discounted Return Bound under Shared-State Topology Approximation.

\[
\boxed{\text{shared-state coupling only}}
\]

Under A1–A3 and a **common** state sequence \(\{s_t\}\),
\[
\bigl|J_T^\star-J_T\bigr|
\le
\frac{L_R L_\pi\varepsilon_G}{1-\gamma}.
\]

**Does not claim** closed-loop \(\rho^\star\approx\rho\) or transition regularity. We do **not** enlarge the claim in this supplement.

Full write-up: [`proofs/thm2_shared_state.md`](proofs/thm2_shared_state.md).  
Lemma 1 (action gap) is included in the same note.

---

### S1.4 Proposition (Communication Complexity)

If \(\max_i d_i\le K\), then \(\lvert E\rvert\le NK\), hence \(C=O(N)\) when \(K=O(1)\).  
Not a performance-generalization theorem (Thm.~3 cancelled).

Full write-up: [`proofs/prop_complexity.md`](proofs/prop_complexity.md).

---

## S2. Algorithm Details

See [`implementation_details/`](implementation_details/).

### S2.1 Topology Projection

Inputs: scores \(S_t=\phi_\theta(s_t)\), candidate support \(A_t\), budget \(B_t\) or degree \(K\).  
Operator: \(G_t=\Pi_{B_t}(S_t)\).  
AC-DSGF realization: row-wise Top-\(K\) (degree budget).  
**No new algorithms** in this supplement.

### S2.2 Training Procedure

Joint optimization of \((\phi_\theta,\pi_\psi)\) on the frozen AC-DSGF + MAPPO backbone under a soft communication surrogate during training; **hard** \(\Pi_{B_t}\) at evaluation. No additional unpublished tricks.

---

## S3. Experimental Details

See [`implementation_details/experimental_settings.md`](implementation_details/experimental_settings.md).

Highlights:
- Primary task T1: cooperative navigation, \(N=16\) formal tables; scaling \(N\in\{16,32,64,128\}\).
- Seeds: \(\{1234,2026,3407,42,8888\}\).
- Method-aware \(C\): MAPPO \(=0\); Full Attention \(=N(N-1)\); GAT/DSGF = radius edges; AC-DSGF = projected edges.

---

## S4. Additional Results

See [`additional_results/README.md`](additional_results/README.md).

Only frozen evidence expansions (grids, CSVs, reports). **No new experiment directions.**

---

## S5. Reproducibility Checklist

See [`reproducibility_checklist.md`](reproducibility_checklist.md).
