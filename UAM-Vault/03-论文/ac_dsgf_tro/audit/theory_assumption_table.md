# Theory Assumption ↔ Experiment Alignment

| Theory | Core assumption(s) | Conclusion (bounded) | Experiment evidence | Must **not** claim |
|--------|--------------------|----------------------|---------------------|--------------------|
| **Thm.~1** | \(\Pi_{B_t}\) solves constrained argmax; budget \(B_t\) or degree \(K\) | \(C(G_t)\le B_t\) (or \(\lvert E(i)\rvert\le K\)); utility-max under scores | §6.2: \(V_{\max}=VR=0\) | Task-optimal \(G\); soft training alone ⇒ hard \(B\) |
| **Lem.~2** | Message map Lipschitz / bounded gap vs reference | \(\varepsilon_G\le L_M\sup d_G\) (bridge) | §6.3 Fig.~1: \(D_G\to\varepsilon_G\) trend | \(\varepsilon_G\) task-optimal or vanishing |
| **Lem.~1** | Policy Lipschitz in message (A2); shared state | \(\|a^\star-a\|\le L_\pi\varepsilon_G\) | §6.3 Fig.~2: \(\varepsilon_G\to\Delta A_\gamma\) | Closed-loop action laws equal |
| **Thm.~2** | Shared-state coupling; A1–A3 | \(\lvert J^\star-J\rvert\le\frac{L_R L_\pi\varepsilon_G}{1-\gamma}\) | Twin (E1) evaluation; Fig.~2 intermediate | **Closed-loop** performance guarantee / \(\rho^\star\approx\rho\) |
| **Cor.~1** | Residual correction gap \(\varepsilon_\Delta\) | Extended action / return bounds | Implementation note | Residual “fixes” all topology loss |
| **Prop.** | Fixed degree \(\max_i d_i\le K\) | \(\lvert E\rvert\le NK\Rightarrow C=O(N)\) if \(K=O(1)\) | §6.4: \(C_N/N\approx1.5\)–\(1.9\) | \(J_N\) preservation; swarm-size generalization |
| **Thm.~3** | — | — | — | **Cancelled** |

---

## Thm.~2 boundary (locked)

**Correct framing:** shared-state topology approximation / twin evaluation.  
**Forbidden framing:** closed-loop performance guarantee.

Manuscript §5 and §6.3 already state this; v0.11 preserves it.

---

## Assumption explicitness check

| File | Assumptions explicit? | Conclusion not enlarged? |
|------|----------------------|--------------------------|
| `theorem_budget.md` | Yes | Yes |
| `lemma_information_discrepancy.md` | Yes | Yes |
| `theorem_return_bound.md` | A1–A3 + “not assumed” list | Yes |
| `proposition_complexity.md` | Degree cap only | Yes |
| `theorem_scalability.md` | N/A (cancelled) | N/A |
