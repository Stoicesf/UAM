# S1.1 — Proof of Theorem 1 (Supplementary)

**Canonical source:** [`../../theory/theorem_budget.md`](../../theory/theorem_budget.md)  
**Main-text title:** Feasibility of Budget-Constrained Dynamic Topology Projection

---

## Setup

- Scores \(S_t\) from topology policy \(\phi_\theta\); gates \(g_{ij}=\sigma(s_{ij})A_{t,ij}\).  
- Feasible family \(\mathcal{G}_{B_t}=\{G:C(G)\le B_t\}\), \(C(G)=\sum_{ij}x_{ij}\).  
- Projection operator
\[
\Pi_{B_t}(S)
=
\arg\max_{x}
\sum_{ij} g_{ij}\,x_{ij}
\quad\mathrm{s.t.}\quad
\sum_{ij}x_{ij}\le B_t,\;
x_{ij}\in\{0,1\},\;
x_{ij}\le A_{t,ij}.
\]
- Degree specialization: \(\sum_j x_{ij}\le K\) per agent (Top-\(K\)).

---

## Statement

For \(G_t=\Pi_{B_t}(S_t)\):

1. **Feasibility:** \(C(G_t)\le B_t\) (or \(\lvert E_t(i)\rvert\le K\)).  
2. **Utility-maximizing projection (not task optimality):** \(G_t\) maximizes \(\sum g_{ij}x_{ij}\) over \(\mathcal{G}_{B_t}\).  
   **Not claimed:** \(G_t\in\arg\max_{G\in\mathcal{G}_{B_t}} J(\pi_\psi,G)\).

---

## Proof

**(1)** Immediate from the constraint set of \(\Pi_{B_t}\).

**(2)** By definition of the constrained argmax.

**Degree-separable case.** The linear objective separates across agents. For each \(i\), retaining the \(K\) largest admissible \(g_{ij}\) is optimal under \(\sum_j x_{ij}\le K\).

□

---

## Operator vs training constraint

Soft \(\lambda c(G)\) in training is a Lagrangian relaxation.  
**Hard feasibility is enforced by executing \(\Pi_{B_t}\)** at decision time / evaluation—not by claiming that SGD alone keeps \(C\le B_t\).
