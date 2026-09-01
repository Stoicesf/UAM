# IEEE Reviewer Risk Scan (pre–mock-review)

Status: hardening pass · no new training  
Updated: 2026-07-15

## Positioning (sell this, not Success↑)

> UAV swarm **communication topology is a learnable decision variable**, optimized under an explicit budget jointly with the policy.

---

## Reviewer 1 — Algorithm

| Risk | Severity | Mitigation now | Artifact |
|------|----------|----------------|----------|
| “Just GAT + penalty / DSGF+ℓ₂?” | High | IV-D comparison table + “topology as decision variable” | `method.tex` §diff-comm |
| “Gate is random dropout” | High | Trigger (risk corr 0.84) + Gate stability Var(C), ΔE | `fig_gate_stability.png` |
| “Why not post-hoc prune?” | Med | Joint obj Eq. joint-max; Gate→Budget→Policy order | `algorithm.tex` |
| Math / symbol inconsistency | Med | Locked notation sheet | `notation.tex` |

**Must-not claim:** AC-DSGF is mainly a Success improver.

---

## Reviewer 2 — Experiments

| Risk | Severity | Mitigation now | Artifact |
|------|----------|----------------|----------|
| Success not clearly better | **Highest** | Retitle: *Task Performance under Communication Constraints*; caption on comparable + reduced Comm | `experiments.tex` Table I |
| “What is Comm=0.43?” | High | Explicit soft-mass definition for AC; edge-count for GAT/DSGF | `problem_definition.tex` Eq.comm-cost |
| Soft cost ≠ radio packets | Med | Limitations note already; keep “orders of magnitude” wording | Limitations |
| Mix 4-UAV ablation with 16-UAV | High | Explicit scale warnings | Table II caption |

**Do not add** Generalization as a main claim (weak advantage).

---

## Reviewer 3 — Systems / engineering

| Risk | Severity | Mitigation now | Deferred |
|------|----------|----------------|----------|
| Complexity not practical | Med | Complexity table + Fig scale | — |
| Packet loss / radio realism | Med–High for RA-L | Mention in Conclusion as future | Optional later |
| Demo ≠ evidence | Low | Demo marked qualitative | — |

---

## Checklist before mock review

- [x] IV-D Difference from conventional comm learning  
- [x] Algorithm 1 Gate→Budget→Policy  
- [x] Cost definition / 0.43 explained  
- [x] Table I retitled & caption fixed  
- [x] Exp section titles A–F style  
- [x] Gate stability figure (eval-only)  
- [x] refs.bib expanded (~30)  
- [ ] Compile v1.pdf + citation polish  
- [ ] Caption language global sweep  
- [ ] Mock reviews → `review_response_mock.md`

## Packet loss

**Defer.** Main story is communication *efficiency*, not robustness. Add only if targeting RA-L and spare time.
