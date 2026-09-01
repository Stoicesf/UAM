# S1.4 — Proposition: Communication Complexity (Supplementary)

**Canonical source:** [`../../theory/proposition_complexity.md`](../../theory/proposition_complexity.md)  
**Not Theorem 3.** Strong graph-size performance / generalization theorem is **cancelled**.

---

## Statement

If \(G=\Pi_B(S)\) satisfies \(\max_i d_i(G)\le K\), then
\[
\lvert E(G)\rvert\le NK,
\qquad
C(G)=O(NK).
\]
When \(K=O(1)\), \(C(G)=O(N)\).

---

## Proof

Each agent contributes at most \(K\) edges; sum over \(N\) agents. □

---

## Empirical companion

§6.4: under fixed \(K=4\), AC-DSGF \(C_N/N\approx 1.5\)–\(1.9\) for \(N:16\to128\).  
\(J_N\) is **not** interpreted as performance preservation.
