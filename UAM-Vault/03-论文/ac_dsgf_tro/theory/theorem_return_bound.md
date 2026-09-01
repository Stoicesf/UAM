# Theorem 2 — Discounted Return Bound under Shared-State Topology Approximation

**Paper:** AC_DSGF_TRO · Phase 3 · **v0.2**  
**Status:** Formal v0.2 · **frozen** · merged EN/CN §5.3 · bridged by Lemma 2
**Depends on:** EN v0.2 symbol freeze · Theorem 1 (`theorem_budget.md`)  
**Official title:** Discounted Return Bound under Shared-State Topology Approximation  
**Forbidden titles:** Return Preservation; Closed-Loop Distributional Equivalence  

**Chain:**
\[
\boxed{
\text{Topology discrepancy}
\;\rightarrow\;
\text{Message discrepancy}
\;\rightarrow\;
\text{Action discrepancy}
\;\rightarrow\;
\text{Reward discrepancy}
}
\]

Complement with Theorem 1 and Lemma 2:
\[
\boxed{\text{Feasibility (Thm.~1)}+\text{Bounded }\varepsilon_G\text{ (Lem.~2)}+\text{Shared-state stability (Thm.~2)}}
\]

**Bridge:** [`lemma_information_discrepancy.md`](lemma_information_discrepancy.md) instantiates A1 via \(\varepsilon_G\le L_M\|A-A^\star\|_F\).

---

## 0. Setup

- Joint policy \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\).  
- Message map \(M\) (write \(M(G)\) when state is clear).  
- **Sparse / decided graph** (Theorem 1):
  \[
  G_t=\Pi_{B_t}\bigl(\phi_\theta(s_t)\bigr),\qquad
  m_t=M(G_t),\qquad
  a_t=\pi_\psi(s_t,m_t).
  \]

### Reference: full-support topology before budget projection (not an optimum)

**Definition (Full-support reference topology before budget projection).**  
\[
G_t^\star
=
\text{pre-projection scores }S_t\text{ on geometric support }A_t
\]
i.e.\ the topology **before** \(\Pi_{B_t}\), not unrestricted all-to-all / infinite communication.

**Wording lock.** Prefer *full-support reference topology before budget projection*.  
**Forbidden:** “full communication graph,” “full optimal graph,” “optimal topology,” or
\[
G_t^\star\in\arg\max_{G} J(\pi_\psi,G).
\]
That conflicts with Theorem 1’s claim boundary.

**Twin actions on a shared state** (same \(\pi_\psi\); topologies differ):
\[
m_t^\star=M(G_t^\star),\qquad
a_t^\star=\pi_\psi(s_t,m_t^\star).
\]

### Topology-induced information discrepancy (primitive)

\[
\varepsilon_G(t)
\;:=\;
\bigl\|M(G_t^\star)-M(G_t)\bigr\|,
\qquad
\varepsilon_G
\;:=\;
\sup_{t\ge 0}\,\varepsilon_G(t).
\]
This is the **primary** discrepancy object. Theorem 2 works directly with \(\varepsilon_G\); it does not require deriving \(\varepsilon_G\) from a graph metric.

### Shared-state (twin) trajectory coupling

Fix a **common** state sequence \(\{s_t\}_{t\ge 0}\) (twin simulation / shared-state evaluation). Define discounted cumulative rewards
\[
J_T^\star
=
\sum_{t=0}^{T}
\gamma^t\,r(s_t,a_t^\star),
\qquad
J_T
=
\sum_{t=0}^{T}
\gamma^t\,r(s_t,a_t),
\]
and infinite-horizon limits \(J^\star=\lim_{T\to\infty}J_T^\star\), \(J=\lim_{T\to\infty}J_T\) when \(\gamma\in[0,1)\).  
Expectations, if used, are over exogenous randomness **conditional on the shared** \(\{s_t\}\)—not over independently rolled-out closed-loop state laws \(\rho_{\pi^\star}\) vs.\ \(\rho_{\pi}\).

---

## 1. Assumptions

### Assumption A1 — Bounded topology-induced message discrepancy

The message aggregation operator induces a finite topology–message gap relative to the full-information reference:
\[
\bigl\|M(G_t^\star)-M(G_t)\bigr|
\;\le\;
\varepsilon_G
\quad\text{for all }t\text{ of interest}.
\]
**Optional metric (not required for Thm.~2).** If a graph distance is needed for exposition,
\[
d_G(G_1,G_2)=\|A_1-A_2\|_F
\]
with adjacency matrices \(A_1,A_2\), and if \(M\) is \(L_M\)-Lipschitz w.r.t.\ \(d_G\), then \(\varepsilon_G(t)\le L_M\,d_G(G_t^\star,G_t)\).  
**We do not rely on this implication in the proof of Theorem 2**—\(\varepsilon_G\) is taken as given.

### Assumption A2 — Policy Lipschitzness

\[
\bigl\|\pi_\psi(s,m_1)-\pi_\psi(s,m_2)\bigr|
\;\le\;
L_\pi\,\|m_1-m_2\|.
\]

### Assumption A3 — Reward Lipschitzness

\[
\bigl|r(s,a_1)-r(s,a_2)\bigr|
\;\le\;
L_R\,\|a_1-a_2\|.
\]

**Not assumed:** RL convergence; topology learning optimality; GNN UA; **transition Lipschitzness / \(\rho_{\pi^\star}\approx\rho_{\pi}\)** (explicitly out of scope).

---

## 2. Lemma 1 — Topology-Induced Action Discrepancy

**Lemma 1.** Under A2 and shared state \(s_t\),
\[
\boxed{
\bigl\|a_t^\star-a_t\bigr|
\;\le\;
L_\pi\,\varepsilon_G(t)
\;\le\;
L_\pi\,\varepsilon_G.
}
\]

**Proof.**  
\(a_t^\star=\pi_\psi(s_t,m_t^\star)\), \(a_t=\pi_\psi(s_t,m_t)\), and \(\|m_t^\star-m_t\|=\varepsilon_G(t)\). Apply A2.  
□

---

## 3. Theorem 2 — Discounted Return Bound under Shared-State Topology Approximation

**Theorem 2 (Discounted return bound under shared-state topology approximation).**  
Under A1–A3, shared-state coupling \(\{s_t\}\), and \(\gamma\in[0,1)\),
\[
\boxed{
\bigl|J_T^\star-J_T\bigr|
\;\le\;
\sum_{t=0}^{T}
\gamma^t\,L_R L_\pi\,\varepsilon_G
\;\le\;
\frac{L_R L_\pi\,\varepsilon_G}{1-\gamma},
}
\]
and likewise \(\lvert J^\star-J\rvert\le L_R L_\pi\varepsilon_G/(1-\gamma)\) in the infinite-horizon limit.

**Proof.**  
By Lemma 1 and A3, at each \(t\),
\[
\bigl|r(s_t,a_t^\star)-r(s_t,a_t)\bigr|
\;\le\;
L_R\|a_t^\star-a_t\|
\;\le\;
L_R L_\pi\,\varepsilon_G.
\]
Sum \(\gamma^t\) from \(t=0\) to \(T\) (and let \(T\to\infty\)).  
□

### Interpretation

Proves: communication **topology** discrepancy (via \(\varepsilon_G\)) causes a **bounded discounted reward discrepancy** under shared-state evaluation.  
Does **not** prove: closed-loop occupancy measures satisfy \(\rho_{\pi^\star}\approx\rho_{\pi}\).

### Remark (claim honesty — retain)

The bound is for **discounted cumulative reward under fixed trajectory coupling** (twin / shared-state). Closed-loop state-distribution drift requires transition Lipschitzness and is **not claimed**.

### Remark (link to Theorem 1)

Thm.~1 \(\Rightarrow\) budget-feasible utility-maximizing projection.  
Thm.~2 \(\Rightarrow\) shared-state return discrepancy controlled by \(\varepsilon_G=\|M(G_{\mathrm{full}})-M(G_t)\|\).  
Score optimality from Thm.~1 is **not** used in Thm.~2.

---

## 4. Corollary — Residual Recovery Bound

**Corollary.** If
\[
a_t=\pi_\psi(s_t,m_t)+\beta\Delta_t,
\qquad
a_t^\star=\pi_\psi(s_t,m_t^\star)+\beta\Delta_t^\star,
\]
with \(\varepsilon_\Delta=\sup_t\|\Delta_t^\star-\Delta_t\|\), then
\[
\boxed{
\bigl\|a_t^\star-a_t\bigr|
\;\le\;
L_\pi\,\varepsilon_G+\beta\,\varepsilon_\Delta
}
\]
and under the same shared-state scope,
\[
\boxed{
\bigl|J^\star-J\bigr|
\;\le\;
\frac{L_R\bigl(L_\pi\,\varepsilon_G+\beta\,\varepsilon_\Delta\bigr)}{1-\gamma}.
}
\]

**Proof.** Triangle inequality on the residual decomposition; then Theorem 2’s discounted sum with the enlarged action gap.  
□

**Wording lock.** Do **not** write \(\beta L_\pi\varepsilon_G\) unless \(\Delta\) is separately shown to absorb the policy-Lipschitz message gap. The additive form \(L_\pi\varepsilon_G+\beta\varepsilon_\Delta\) is required.

**Special case.** If \(\Delta\) is \(L_\Delta\)-Lipschitz in the message and \(\varepsilon_\Delta\le L_\Delta\varepsilon_G\), then
\[
|J^\star-J|
\le
\frac{L_R(L_\pi+\beta L_\Delta)\,\varepsilon_G}{1-\gamma}.
\]

---

## 5. Interpretation — Proved / Not Proved

| Proved | Not proved |
|--------|------------|
| Shared-state \(\lvert J^\star-J\rvert\le L_R L_\pi\varepsilon_G/(1-\gamma)\) | “Return preservation” / performance never decreases |
| Action gap \(\le L_\pi\varepsilon_G\); residual \(\le L_\pi\varepsilon_G+\beta\varepsilon_\Delta\) | \(\rho_{\pi^\star}\approx\rho_{\pi}\) (closed-loop drift) |
| \(G^\star=G_{\mathrm{full}}\) as **reference**, not optimum | \(G_t=\arg\max J\); soft \(\lambda\) alone bounds \(\varepsilon_G\) |
| Topology \(\rightarrow\) message \(\rightarrow\) action \(\rightarrow\) reward | RL convergence; channel / GNN-UA theorems |

**One-sentence claim for §5:**  
Under shared-state evaluation, if topology-induced information discrepancy stays bounded, discounted cumulative reward discrepancy remains controllable.

---

## 6. Merge checklist

- [x] Title = shared-state topology approximation (not “preservation”)  
- [x] \(G^\star=G_{\mathrm{full}}\) = reference, not optimum  
- [x] Residual = \(L_\pi\varepsilon_G+\beta\varepsilon_\Delta\)  
- [x] Twin-trajectory \(J_T^\star,J_T\) explicit  
- [x] Inline condensed §5 in EN/CN (v0.3 merge)
- [x] Do not open Theorem 3 / experiments in the same merge
