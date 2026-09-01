# Final Reviewer Simulation (pre-submit)

**Status:** Submission Candidate · scientific freeze · Class C deferred

---

## Reviewer #1 — Decision vs pruning

**Q:** Is topology really a decision variable, or only pruning?

**A (locked):**  
It is an online decision \(G_t=\phi_\theta(s_t)\) inside a DGDP, followed by hard projection \(\Pi_{B_t}\) onto \(\mathcal{G}_{B_t}\). This is **not** post-training / post-hoc pruning of a fixed graph. See §1.1 boxed claim, Algorithm 1, Theorem 1, and §7.

---

## Reviewer #2 — Closed-loop theory

**Q:** Does theory guarantee closed-loop performance?

**A (locked):**  
**No.** Theorem 2 is a **shared-state** discounted return bound under twin coupling. Closed-loop guarantees would require additional transition regularity; we do not add that theorem.

---

## Reviewer #3 — Scalability

**Q:** Does \(N=128\) prove scalability / generalization?

**A (locked):**  
**No.** Under fixed degree, the complexity proposition and §6.4 support \(C_N=O(N)\). \(J_N\) is observational under changing task scale/difficulty—not performance preservation. Strong Theorem 3 remains cancelled.

---

## Package stance on Class C

If asked why TarMAC/IC3Net are absent: we compare **communication-cost regimes** along the topology-decision axis; Class C is optional supplementary and not required for the core claim.
