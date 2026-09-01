# S1.3 — Proof of Theorem 2 (Supplementary)

**Canonical source:** [`../../theory/theorem_return_bound.md`](../../theory/theorem_return_bound.md)  
**Official title:** Discounted Return Bound under Shared-State Topology Approximation

\[
\boxed{\text{shared-state coupling only}}
\]

**Forbidden enlargement in this supplement:** closed-loop performance guarantee; \(\rho^\star\approx\rho\); transition theorems.

---

## Shared-state coupling

Fix a common state sequence \(\{s_t\}\). Twin actions:
\[
a_t^\star=\pi_\psi(s_t,M(G_t^\star)),
\qquad
a_t=\pi_\psi(s_t,M(G_t)).
\]
\[
\varepsilon_G(t)=\|M(G_t^\star)-M(G_t)\|,
\qquad
\varepsilon_G=\sup_t\varepsilon_G(t).
\]

---

## Assumptions

- **A1:** \(\varepsilon_G<\infty\) (optionally via Lemma 2).  
- **A2:** \(\|\pi_\psi(s,m_1)-\pi_\psi(s,m_2)\|\le L_\pi\|m_1-m_2\|\).  
- **A3:** \(\lvert r(s,a_1)-r(s,a_2)\rvert\le L_R\|a_1-a_2\|\).  

**Not assumed:** transition Lipschitzness; RL / topology-learning convergence.

---

## Lemma 1 (action discrepancy)

\[
\|a_t^\star-a_t\|\le L_\pi\varepsilon_G(t)\le L_\pi\varepsilon_G.
\]

**Proof.** Apply A2 under shared \(s_t\). □

---

## Theorem 2

For \(\gamma\in[0,1)\),
\[
\bigl|J_T^\star-J_T\bigr|
\le
\sum_{t=0}^{T}\gamma^t L_R L_\pi\varepsilon_G
\le
\frac{L_R L_\pi\varepsilon_G}{1-\gamma}.
\]

**Proof.** By Lemma 1 and A3,
\[
\lvert r(s_t,a_t^\star)-r(s_t,a_t)\rvert
\le
L_R L_\pi\varepsilon_G.
\]
Sum \(\gamma^t\). □

---

## Interpretation (locked)

The bound characterizes **topology-induced degradation under shared-state coupling**.  
Closed-loop guarantees would require additional transition regularity assumptions—which are **not** introduced here or in the main text.
