# Lemma 2 — Topology Sparsification Induces Bounded Information Discrepancy

**Paper:** AC_DSGF_TRO · Phase 3+ (bridge)  
**Status:** Formal · bridges Theorem 1 → Theorem 2  
**Role:** Explain why a budget-projected topology yields a **finite** topology-induced message discrepancy \(\varepsilon_G\).  
**Does not claim:** that \(\varepsilon_G\) is small enough for a target task return; that projection is task-optimal; closed-loop \(\rho^\star\approx\rho\).

**Full chain:**
\[
\boxed{
\text{Budget (Thm.~1)}
\;\rightarrow\;
\text{Topology distance (Lem.~2)}
\;\rightarrow\;
\text{Information loss}
\;\rightarrow\;
\text{Return bound (Thm.~2)}
}
\]

---

## 0. Setup

From Theorem 1:
\[
G_t=\Pi_{B_t}(S_t)=\Pi_{B_t}\bigl(\phi_\theta(s_t)\bigr)
\in\mathcal{G}_{B_t},
\qquad
C(G_t)\le B_t.
\]

**Full-support reference topology before budget projection** (not an optimum, not unrestricted all-to-all):
\[
G_t^\star
=
\text{pre-projection scores on geometric support }A_t
\quad\text{(denoted }G_{\mathrm{full\text{-}support}}\text{)}.
\]

Let \(A_t\) and \(A_t^\star\) be the adjacency matrices of \(G_t\) and \(G_t^\star\). Define the **graph distance**
\[
d_G(G_t,G_t^\star)
\;:=\;
\|A_t-A_t^\star\|_F.
\]
Since edge indicators are binary and \(G_t\subseteq G_t^\star\) under standard sparsification of a full support (dropped edges only),
\[
d_G(G_t,G_t^\star)^2
=
\sum_{ij}(A_{t,ij}^\star-A_{t,ij})^2
=
\bigl|E(G_t^\star)\setminus E(G_t)\bigr|,
\]
hence \(d_G\) is finite whenever \(N<\infty\) and \(A_t^\star\) has finitely many ones.

Message discrepancy (Theorem 2 primitive):
\[
\varepsilon_G(t)=\bigl\|M(G_t^\star)-M(G_t)\bigr\|,
\qquad
\varepsilon_G=\sup_t\varepsilon_G(t).
\]

---

## 1. Assumption (message map regularity)

**Assumption M (Message Aggregation Lipschitzness w.r.t.\ topology).**  
There exists \(L_M\ge 0\) such that for all admissible graphs \(G_1,G_2\) and shared state \(s\),
\[
\bigl\|M(G_1;s)-M(G_2;s)\bigr|
\;\le\;
L_M\,d_G(G_1,G_2)
=
L_M\,\|A_1-A_2\|_F.
\]
**Intent:** keep the object in topology space. Not a packet-loss / Shannon channel model.

*(Standard graph aggregators that are linear—or Lipschitz—in adjacency-weighted neighbor features satisfy Assumption M with \(L_M\) depending on feature norms and aggregation weights.)*

---

## 2. Lemma 2

**Lemma 2 (Topology sparsification induces bounded information discrepancy).**  
Let \(G_t=\Pi_{B_t}(S_t)\) be the budget-constrained projection of Theorem 1, and let \(G_t^\star\) be the **full-support reference topology before budget projection**. Under Assumption M,
\[
\boxed{
\varepsilon_G(t)
=
\bigl\|M(G_t^\star)-M(G_t)\bigr|
\;\le\;
L_M\,d_G(G_t,G_t^\star)
=
L_M\,\|A_t-A_t^\star\|_F.
}
\]
In particular, for finite \(N\) and finite support,
\[
\varepsilon_G
\;\le\;
L_M\sup_t d_G(G_t,G_t^\star)
\;<\;
\infty.
\]

**Proof.**  
Apply Assumption M with \(G_1=G_t^\star\), \(G_2=G_t\). Finiteness follows because \(d_G\le\sqrt{|E(G_t^\star)|}\le N\) (or \(\sqrt{N(N-1)}\) for directed complete support).  
□

### Remark (what Lemma 2 does / does not do)

| Does | Does not |
|------|----------|
| Bridge Thm.~1’s feasible \(G_t\) to Thm.~2’s \(\varepsilon_G\) | Claim \(\varepsilon_G\) is optimally small |
| Show projection \(\Rightarrow\) **finite** message gap under Lipschitz \(M\) | Replace Thm.~2’s shared-state scope |
| Allow writing \(\varepsilon_G\le L_M\|A-A^\star\|_F\) in §5 | Claim \(d_G\to 0\) as learning proceeds |

Soft \(\lambda\)-training and score quality affect *which* edges \(\Pi_{B_t}\) keeps (hence the size of \(d_G\)); Lemma 2 only transfers that graph distance into message space.

### Remark (degree-budget specialization)

Under per-agent Top-\(K\) projection, at most \(NK\) edges are kept. If \(|E(G_t^\star)|=N(N-1)\) (directed full) or \(\binom{N}{2}\) (undirected), then
\[
d_G(G_t,G_t^\star)^2
=
|E(G_t^\star)|-|E(G_t)|
\ge
|E(G_t^\star)|-NK,
\]
so the gap can grow with \(N\) unless \(K\) or the support is scaled—**this motivates future scaling experiments**, not an automatic Theorem 3.

---

## 3. Interface to Theorem 2

Theorem 2 may take \(\varepsilon_G\) as given (A1), or instantiate
\[
\varepsilon_G
\;\le\;
L_M\sup_t\|A_t-A_t^\star\|_F
\]
via Lemma 2. Combined with Lemma 1 (action gap) and Theorem 2 (return bound):
\[
|J_T^\star-J_T|
\;\le\;
\frac{L_R L_\pi\varepsilon_G}{1-\gamma}
\;\le\;
\frac{L_R L_\pi L_M}{1-\gamma}
\sup_t d_G(G_t,G_t^\star).
\]

---

## 4. Consistency

- [x] \(G^\star=G_{\mathrm{full}}\) is reference, not \(\arg\max J\)  
- [x] Uses \(G_t=\Pi_{B_t}(\phi_\theta(s_t))\) from Theorem 1  
- [x] Feeds \(\varepsilon_G\) into Theorem 2 without claiming closed-loop equivalence  
- [x] No scalability / \(N\to\infty\) theorem here
