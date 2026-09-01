# AC-DSGF v1 — Submission Readiness Checklist

**Status:** Theory & Evidence Alignment revision applied (2026-07-18).  
**Master constraints:** [`LANGUAGE_FREEZE.md`](LANGUAGE_FREEZE.md) · [`THEORY_EVIDENCE_ALIGNMENT.md`](THEORY_EVIDENCE_ALIGNMENT.md)

## Alignment checklist (this revision)

- [x] Prop.1 = degree-constraint **inequality** \(\eta_N\le K/(N-1)=\mathcal{O}(1/N)\)
- [x] Prop.2 = Residual **action** stability (not task-preservation guarantee)
- [x] Prop.3 = Topology optimality interpretation of gates
- [x] SCA defined once; Soft Mass terminology retired in main text
- [x] Algorithm 1 expanded (score → relax → budget → aggregate → residual)
- [x] Table 1 vs Table 2 protocol separated
- [x] AC-random narrative rewritten (Success ≠ efficiency)
- [x] Hard Top-K Table 2b / Fig.14 added (**honest:** Distance ≈ AC > Random)
- [x] CN manuscript rebuilt after encoding corruption (UTF-8 verified)

---

## Phase A — Theory consistency

- [x] Prop.1 title = **Communication Scalability Bound**
- [x] Prop.1 first-line intent: scalability / bounded growth / sparsification under constraints
- [x] No *learning/optimization convergence* claim in Abstract / Conclusion / Prop.1
- [x] Objective = task-aware topology selection; Constraint = communication budget  
  (not “maximize performance while minimizing communication”)

**Keyword scan (main MD):** `converge*`, `optimal*`, `guarantee`, `prove` → only conditional/disclaimer uses remain.

---

## Phase B — Method consistency

- [x] Stage names frozen everywhere:
  1. Candidate Edge Scoring  
  2. Budget-Constrained Edge Selection  
  3. Residual Recovery  
- [x] Algorithm 1 title: **Constraint-aware Adaptive Topology Selection**
- [x] Prefer *stage / procedure* over *module / block / component*

---

## Phase C — Experiment claims

- [x] Headline: *Under identical communication budgets, AC-DSGF achieves task-aware topology selection with competitive task performance.*
- [x] No primary claim “reduces communication” when \(|E_{\mathrm{AC}}|=|E_{\mathrm{RULE}}|\)
- [x] Soft Mass / dense-open comparison = auxiliary proxy only
- [x] Caption policy: *Topology Selection / Adaptation under Communication Constraints*  
  (not Communication Reduction / Bandwidth Saving)

---

## Phase D — Contributions (3 only)

1. Constrained optimization formulation of topology selection  
2. AC-DSGF three-stage framework (task relevance + budgets)  
3. Scalability analysis + multi-scale / multi-condition evaluations  

Out of scope: SwarmOS, PX4, engineering framework as paper contributions.

---

## Phase E — Engineering isolation

Paper may mention:

- simulation platform  
- hardware-compatible evaluation  

Paper must not claim:

- SwarmOS / Runtime / Adapter / SafetySupervisor / RecoveryManager as algorithmic contributions  

---

## Files to submit (Markdown-first)

| Item | Path |
|------|------|
| EN manuscript | `AC_DSGF_EN.md` |
| CN manuscript | `../ac_dsgf_cn/AC_DSGF_CN.md` |
| Cover | `COVER_LETTER.md` / `COVER_LETTER_CN.md` |
| Highlights | `HIGHLIGHTS.md` / `HIGHLIGHTS_CN.md` |
| Contributions | `CONTRIBUTION_STATEMENT.md` / `CONTRIBUTION_STATEMENT_CN.md` |
| Figs | `figures/` + `../ac_dsgf_cn/figures/` |

---

## Final story (reviewer view)

```
Problem:   topology usually predefined
Insight:   topology should be optimized under budgets
Method:    AC-DSGF constrained task-aware selection (3 stages)
Theory:    Communication Scalability Bound
Experiment: identical budgets → competitive effectiveness + adaptation
```

**Do next:** only language-level fixes if a residual keyword is found. Do not extend the system or reopen ++.

## RA-L final gate (2026-07-18)

- [x] Table 2: AC-random vs AC-full SCA **~2,600×** callout + near-zero activation thesis sentence
- [x] Figure display numbers consecutive 0–11 (filenames may keep historical prefixes)
- [x] Response mock refreshed: Q1 hard Top-K train / Q2 random Success / Q3 Prop.1 tightness → 
esponse_mock.md
- [ ] Optional camera-ready: multi-λ short train appendix only (do not rewrite Fig.10 narrative)
