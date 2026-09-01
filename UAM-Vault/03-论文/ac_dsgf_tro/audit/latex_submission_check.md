# LaTeX / Submission Formatting Check — v0.11

**Scope:** package formatting readiness. Scientific content frozen.  
**Source of truth (now):** Markdown EN/CN. BibTeX: `references/ac_dsgf_tro.bib`.

---

## Manuscript

### Title

| Check | Status |
|-------|--------|
| Current: *… Coordination Variable under Resource Constraints* | ✅ |
| No “scalable / universal / optimal” in title | ✅ |
| Center object still \(G_t=\phi_\theta(s_t)\) | ✅ |

### Abstract

| Check | Status |
|-------|--------|
| Problem + formulation + theory + empirics | ✅ |
| Core: topology as learnable coordination variable | ✅ |
| Forbidden closing “achieves scalable coordination” | ✅ avoided |
| Preferred empiric close: communication-efficient under constrained budgets | ✅ (v0.11 package polish) |

### Theorem formatting (when ported to LaTeX)

| Item | Requirement | Notes |
|------|-------------|-------|
| Thm.~1 | Assumption / Statement / Proof | `theory/theorem_budget.md` |
| Lem.~1–2 | Statement + assumptions | info + action discrepancy |
| Thm.~2 title | **Discounted Return Bound under Shared-State Topology Approximation** | keep; shared-state only |
| Prop. | Degree cap ⇒ \(C=O(N)\) | not Theorem 3 |
| Each thm | No enlarged conclusion in caption | checklist |

### Figures Fig.~1–8

| Fig | Role | In text? | Standalone claim? |
|-----|------|----------|-------------------|
| 1 | \(D_G\to\varepsilon_G\) | §6.3 | no — “consistent with” Lem.~2 |
| 2 | \(\varepsilon_G\to\Delta A_\gamma\) | §6.3 | no — twin / Lem.~1 |
| 3 | \(B\to(J,C)\) | §6.3 | tradeoff illustration only |
| 4 | \(\rho\) vs \(N\) (budget dens.) | §6.2 | no scalability claim |
| 5 | \((J,C)\) Pareto | §6.6 | cost regimes |
| 6 | channel stress | §6.5 | performance stability, not robustness thm |
| 7 | \(C_N\) vs \(N\) | §6.4 | Obs.~1–2 |
| 8 | \(\eta_N\) vs \(N\) | §6.4 | Obs.~3 efficiency |

Captions for camera-ready should repeat the same hygiene (no “proves / validates / scalable”).

### Tables

| Table | Status |
|-------|--------|
| I Budget feasibility | frozen |
| II Baselines \((J,C)\) | frozen |
| III Scalability stats | frozen |

---

## Class C

**Decision:** deferred. Not required for submission package.  
Risk (impl consistency / budget normalization / training fairness) > benefit.

---

## Supplementary (pending)

| Item | Status |
|------|--------|
| Proof appendices from `theory/` | pending final pack |
| Protocol links / evidence paths | available under `experiments/` |
| Extra figures | not required |
