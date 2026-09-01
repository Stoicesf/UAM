# Mock Reviewer Simulation (pre-submission)

## Reviewer 1 — Theory

**Q:** Why a shared-state / twin bound instead of a closed-loop performance guarantee?

**A (locked wording):**  
The bound characterizes topology-induced degradation under shared-state coupling, while closed-loop guarantees require additional transition regularity assumptions. We intentionally do **not** add a transition theorem: it would enlarge the theory scope and introduce new assumption burden. Empirically, §6.3 uses twin evaluation for \(\Delta A_\gamma\) and reports closed-loop tradeoffs separately (Fig.~3).

---

## Reviewer 2 — Experiments / baselines

**Q:** Why no TarMAC / IC3Net / ATOC?

**A (for rebuttal if asked; do not lead with “not implemented” in the paper):**  
We compare against representative communication paradigms that correspond to different communication-cost regimes. The scientific axis is **topology decision under budget** \(G_t=\phi_\theta(s_t)\) with \(\Pi_{B_t}\), not attention-mechanism competition on a fixed support. Class C remains optional supplementary work (priority TarMAC > IC3Net > ATOC), not a core gap.

---

## Reviewer 3 — Scalability / generalization

**Q:** Does \(N=16\to128\) prove scalability / generalization?

**A (locked):**  
**No.** \(C_N=O(N)\) is supported by the complexity proposition + §6.4 Obs.~1. \(J_N\) is an observation under changing task scale/difficulty, not performance preservation. Strong Theorem 3 is cancelled.
