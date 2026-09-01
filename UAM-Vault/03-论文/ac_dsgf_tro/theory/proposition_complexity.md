# Proposition — Communication Complexity under Fixed-Degree Projection

**Paper:** AC_DSGF_TRO · post–§6.4 decision  
**Official title:** Communication Complexity under Fixed-Degree Projection  
**Status:** **frozen** (weak form; not Theorem 3)  
**Forbidden titles:** Graph Scalability Guarantee; Performance Generalization Theorem  
**Does not claim:** \(J_N\) preservation; transfer from \(N=16\) to \(N=128\); GNN stability; permutation-invariant generalization

---

## Relation to Theorem 3 (cancelled)

A strong **Theorem 3** (graph-size performance / generalization) is **not** written.  
§6.4 supports empirical communication scaling and efficiency, not a closed-loop \(J_N\) scalability theorem.  
This proposition is the theory-facing counterpart: a direct corollary of the degree-budget specialization of Theorem 1.

Legacy stub: [`theorem_scalability.md`](theorem_scalability.md) (cancelled).

---

## Setup

- Agent set \(V=\{1,\ldots,N\}\).  
- After projection \(G=\Pi_{B}(S)\) under a **per-agent degree budget** \(K\in\mathbb{N}\):
  \[
  \max_{i\in V} d_i(G)\le K,
  \]
  where \(d_i(G)=\lvert\{j:(i,j)\in E(G)\}\rvert\) (or the directed out-degree, matching the Top-\(K\) realization).  
- Communication cost (edge count; same accounting as §6.6):
  \[
  C(G)=\lvert E(G)\rvert.
  \]

---

## Statement

**Proposition (Communication complexity under fixed-degree projection).**  
If \(G=\Pi_B(S)\) satisfies \(\max_i d_i(G)\le K\), then
\[
\lvert E(G)\rvert \le NK.
\]
Consequently
\[
C(G)=O(NK).
\]
In particular, when the degree budget is held constant in \(N\) (\(K=O(1)\)),
\[
C(G)=O(N).
\]

---

## Proof

Each agent contributes at most \(K\) incident (or outgoing) edges. Summing over \(N\) agents yields \(\lvert E\rvert\le NK\). The \(O(\cdot)\) statements are immediate.  
□

*(Directed Top-\(K\) rows: \(\lvert E\rvert=\sum_i\lvert E(i)\rvert\le NK\). Undirected graphs with the same degree cap obey the same bound up to the usual factor \(1/2\) in double-counting; the \(O(N)\) rate is unchanged.)*

---

## What this does / does not prove

| Proves | Does not prove |
|--------|----------------|
| Linear edge / cost growth under fixed \(K\) | Task return \(J_N\) stays “stable” as \(N\) grows |
| Consistency with Thm.~1 degree feasibility | Optimal coordination at large \(N\) |
| A clean complexity companion to §6.4 Obs.~1 | Graphon / GNN / transfer theorems |

---

## Link to experiments (§6.4)

Under frozen \(K=4\), empirical \(C_N/N\) for AC-DSGF stays in a narrow band \(\approx 1.5\)–\(1.9\) as \(N:16\to128\), **consistent with** \(C=O(N)\).  
Full Attention (\(C=N(N-1)\)) exhibits near-quadratic growth and is the dense reference—not a counterexample to the proposition (it does not enforce a fixed degree budget).

---

## Manuscript placement

Theory section (after Thm.~2 / Cor.~1): **Proposition** — not numbered as Theorem 3.  
Contribution wording (preferred): *budget-constrained topology learning with linear communication growth under fixed-degree projection.*
