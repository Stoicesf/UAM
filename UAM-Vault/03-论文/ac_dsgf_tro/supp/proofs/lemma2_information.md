# S1.2 — Proof of Lemma 2 (Supplementary)

**Canonical source:** [`../../theory/lemma_information_discrepancy.md`](../../theory/lemma_information_discrepancy.md)

---

## Objects

\[
\varepsilon_G(t)=\bigl\|M(G_t^\star)-M(G_t)\bigr\|,
\qquad
d_G(G_t,G_t^\star)=\|A_t-A_t^\star\|_F.
\]

\(G_t^\star\): **full-support reference topology before budget projection** (not \(\arg\max J\)).  
\(G_t=\Pi_{B_t}(S_t)\).

---

## Assumption M

There exists \(L_M\ge 0\) such that
\[
\bigl\|M(G_1;s)-M(G_2;s)\bigr|
\le
L_M\,d_G(G_1,G_2).
\]

---

## Lemma 2

\[
\varepsilon_G(t)
\le
L_M\,d_G(G_t,G_t^\star)
=
L_M\|A_t-A_t^\star\|_F.
\]
Hence for finite \(N\), \(\varepsilon_G=\sup_t\varepsilon_G(t)<\infty\).

---

## Proof

Apply Assumption M with \(G_1=G_t^\star\), \(G_2=G_t\).  
Finiteness: \(d_G\le\sqrt{\lvert E(G_t^\star)\rvert}<\infty\) for finite support.

□

**Does not claim:** \(\varepsilon_G\) is task-optimal or vanishing; closed-loop \(\rho^\star\approx\rho\).
