# Theorem 1 — Feasibility of Budget-Constrained Dynamic Topology Projection

**Paper:** AC_DSGF_TRO · v0.2 language freeze  
**Official title:** Feasibility of Budget-Constrained Dynamic Topology Projection  
**Forbidden titles:** Topology Optimization Guarantee; Optimal Communication Guarantee  
**Does not claim:** task-optimal topology; \(\max_G J(\pi,G)\); learning convergence

---

## Setup

- Agent set \(V=\{1,\ldots,N\}\).  
- Candidate / geometric support \(A_{t,ij}\in\{0,1\}\).  
- Scores \(S=\{s_{ij}\}\) from **topology policy** \(\phi_\theta\); gates \(g_{ij}=\sigma(s_{ij})\,A_{t,ij}\).  
- Time-varying budget \(B_t\) (static case: \(B_t\equiv B\)).  
- Feasible family \(\mathcal{G}_{B_t}=\{G:C(G)\le B_t\}\) with \(C(G)=\sum_{ij}x_{ij}\).

**Projection operator** (abstraction; no solver commitment)
\[
\Pi_{B_t}(S)
=
\arg\max_{x}
\sum_{ij} g_{ij}\,x_{ij}
\quad
\mathrm{s.t.}
\quad
\sum_{ij}x_{ij}\le B_t,\;
x_{ij}\in\{0,1\},\;
x_{ij}\le A_{t,ij}.
\]

**Degree-budget specialization** (optional realization, e.g.\ AC-DSGF Top-\(K\)): for each \(i\),
\[
\sum_{j}x_{ij}\le K,\quad x_{ij}\le A_{t,ij}.
\]
Then \(G_t=\Pi_{B_t}(S_t)\) may be row-wise Top-\(K\).

---

## Statement

**Theorem 1 (Feasibility of budget-constrained dynamic topology projection).**  
For any score matrix \(S_t\), let \(G_t=\Pi_{B_t}(S_t)\) with indicator \(x^\star\). Then:

1. **Feasibility:** \(C(G_t)\le B_t\) (or \(|E_t(i)|\le K\) for all \(i\) under degree budgets).  
2. **Utility-maximizing projection (not task optimality):** for all \(G'\in\mathcal{G}_{B_t}\) with indicator \(x'\),
   \[
   \sum_{ij}g_{ij}\,x^\star_{ij}
   \ge
   \sum_{ij}g_{ij}\,x'_{ij}.
   \]
   Equivalently: \(G_t\) is a **budget-feasible utility-maximizing projection** under learned edge utilities.  
   **Not claimed:** \(G_t\in\arg\max_{G\in\mathcal{G}_{B_t}} J(\pi_\psi,G)\).

---

## Proof

**(1)** Immediate from the feasible set of \(\Pi_{B_t}\).

**(2)** By definition of the argmax over \(\mathcal{G}_{B_t}\).

**Degree-separable case.** Objective \(\sum_i\sum_j g_{ij}x_{ij}\) separates across \(i\). For each \(i\), the optimum under \(\sum_j x_{ij}\le K\) and \(x_{ij}\le A_{t,ij}\) is to set \(x_{ij}=1\) on the \(K\) largest admissible \(g_{ij}\) (or fewer if fewer candidates exist). Union over \(i\) yields a global optimum for the separable problem.

---

## What this does / does not prove

| Proves | Does not prove |
|--------|----------------|
| Hard budget / degree feasibility after projection | \(G_t\) maximizes task return \(J\) |
| Linear **utility** optimality in \(\mathcal{G}_{B_t}\) | Soft SCA training alone enforces hard \(B_t\) |
| Consistency of Top-\(K\) as one realization of \(\Pi_{B_t}\) | Score \(s_{ij}\) equals true edge value \(\Delta R_{ij}\) |
| | That \(\Pi_{B_t}\) is “the optimal topology” |

Soft training with \(\lambda\,c(G)\) is a Lagrangian *relaxation*; **Theorem 1 applies whenever \(\Pi_{B_t}\) is executed**.

---

## Link to Algorithm 1

Step 2: \(G_t=\Pi_{B_t}(S_t)\). Naming: **constraint projection operator**, not heuristic selection, not solver expansion.

---

## Presentation alternative (narrative; equivalent)

For DGDP-facing prose, prefer graph decision variables:
\[
\Pi_{B_t}(S_t)
=
\operatorname*{argmax}_{G\in\mathcal{G}_{B_t}}
U_\theta(G\mid s_t),
\qquad
U_\theta(G\mid s_t)
=
\sum_{(i,j)\in E(G)} g_{ij}.
\]
The indicator / \(x\)-form above remains the proof-friendly equivalent.
