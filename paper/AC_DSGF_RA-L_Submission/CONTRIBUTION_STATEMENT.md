# Contribution Statement — AC-DSGF (FINAL · RA-L)
# Three contributions only: Formulation · Method · Validation

---

## Manuscript title
Adaptive Communication-Constrained DSGF for Efficient UAV Swarm Coordination

## Summary sentence
We study **communication-efficient cooperative control** for UAV swarms by
formulating adaptive topology selection as a learnable graph decision problem,
designing a budget-aware residual coordination framework, and validating
communication–performance trade-offs at scale.

---

## Contribution 1 — Problem formulation
> We formulate adaptive communication topology selection as a **learnable graph
> decision problem**, rather than treating communication as a distance-defined
> or dense predefined structure.

**Keywords:** problem formulation · adaptive communication topology ·
task-aware sparse interaction

---

## Contribution 2 — Method
> We design a **budget-aware adaptive communication framework** integrating
> gate learning, Top-$K$ budgeting, and residual guidance for stable sparse
> coordination under explicit communication cost
> \(J=\mathbb{E}[R]-\lambda C\), \(C=\frac1T\sum_t\sum_{ij}g_{ij}^{t}\).

**Keywords:** method · communication budget optimization · residual guidance

---

## Contribution 3 — Validation
> We provide extensive evaluations demonstrating that **sparse communication
> can maintain cooperative performance** while **significantly reducing
> communication cost** (16-UAV, multi-seed; budget, ablation, behavior, and
> robustness studies).

**Keywords:** validation · communication-constrained coordination ·
scalable UAV swarm

---

## Explicit non-claims
- Not “superior Success” vs dense-graph DSGF  
- Not causal / outcome-aware communication as a main contribution  
- Soft \(C\) is a communication-demand indicator (threshold/Top-$K$ for deploy)
