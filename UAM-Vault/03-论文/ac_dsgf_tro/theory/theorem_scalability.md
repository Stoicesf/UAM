# Theorem 3 — Dynamic-Graph Scalability / Generalization (**CANCELLED**)

**Status:** **Not written.** Post–§6.4 decision (2026-07-24).

## Decision

Do **not** draft a strong Theorem 3 claiming performance generalization / graph-size guarantees.

**Reasons (evidence vs.\ needed claim):**
- §6.4 supports \(C_N=O(N)\) under fixed \(K\), not performance preservation.
- Episode return \(J_N\) grows with task scale; success rate falls with \(N\) (difficulty shift).
- No permutation-invariance / GNN-stability / transition-distribution bound available.

## Replacement

Use the weak form:

→ [`proposition_complexity.md`](proposition_complexity.md)  
**Proposition (Communication complexity under fixed-degree projection):**  
\(\max_i d_i\le K \Rightarrow \lvert E\rvert\le NK \Rightarrow C=O(N)\) when \(K=O(1)\).

§6.4 remains **empirical scalability analysis** (three observations + Table III + Fig.~7–8).
