# AC-DSGF v1 — Three-Round Mock Review + Revision Checklist
# Paper Hardening Phase · 2026-07-17
# Claim lock: comparable Success + Comm↓ / CEI↑ (NOT Success leadership)

---

## Round A — Reviewer 1

### Attack
> “Is communication reduction just disabling messages / random dropout / DSGF+ℓ₂?”

### Required evidence (already have)
| Item | Where |
|------|--------|
| Learned gate vs random / full local | Table silence (AC-full / no-budget / random) |
| Gate non-chattering | Fig. gate stability (>99% ΔE=0) |
| Joint objective in forward pass | Method § joint + Alg.1 |
| Task-driven sparsity | Behavior analysis (risk corr 0.84) |

### Manuscript must say
> Communication is optimized **as part of** policy learning, not as post-hoc pruning.  
> Silence ablation: opening all radius edges inflates Comm by >3 orders with only a small Success gain → sparsity ≠ collapse.

### Revision checklist
- [x] Related Work Gap~3: $P(\mathrm{drop})$ vs $\mathrm{learn}(g_{ij})$
- [x] Method “Difference from pruning” paragraph
- [ ] Caption of silence table: explicit “not silence collapse”
- [ ] Cover letter one sentence on joint vs post-hoc

---

## Round B — Reviewer 2

### Attack
> “Sparse communication hurts task performance / Success not higher than DSGF.”

### Required evidence
| Item | Where |
|------|--------|
| Success hold | Table I: 3.95±0.83 vs DSGF 4.01±1.68 |
| Comm / CEI | 0.43 vs 39.27; CEI 0.091 vs 0.001 |
| Budget graceful degradation | Fig. budget / Table budget |
| Residual prevents degeneration | Ablation w/o residual 0.22 → Full 9.27 (N=4 mech.) |

### Manuscript must say
> AC-DSGF **maintains comparable task performance** while reducing communication by ~**two orders of magnitude**.  
> We **do not** claim Success leadership.

### Revision checklist
- [x] Overall Performance subsection + reading guide (no “outperforms”)
- [x] Abstract / Intro claim-safe
- [ ] Pareto caption: “high-efficiency region” (left = low Comm)
- [ ] Limitations: absolute Success low for *all* methods on this hard bench

---

## Round C — Reviewer 3

### Attack
> “Scalability? Why not just shrink $R_c$? Soft mass ≠ packets?”

### Required evidence
| Item | Where |
|------|--------|
| Complexity $O(N^2)\!\to\!O(NK)$ | Lemma 1 + Sec. complexity |
| Adaptive vs static radius | Behavior analysis + Method |
| Packet-loss flat curve | Fig./Table packet loss |
| Soft-cost definition | Notation / Limitations |

### Manuscript must say
> Radius shrink is geometry-only and static; AC-DSGF opens/closes links by task risk.  
> Soft $C=\sum g_{ij}$ is an intensity proxy; Limitations acknowledge packet mapping.

### Revision checklist
- [x] Theory remarks Lemma 1–2
- [x] Communication Behavior Analysis rename (not “Trigger” only)
- [ ] One sentence in Limitations on soft vs physical packets
- [ ] Optional (Week 2): 32-UAV *eval-only* Comm row (no Success chase)

---

## Preferred rebuttal one-liner
> AC-DSGF learns **who / when / how much** to communicate under a budget while residual guidance keeps messaging assistive—not Success chasing under unlimited radio.

---

## Figure order (recommended for camera-ready)
| Fig | Role |
|-----|------|
| 1 | Framework (gate + budget + residual) |
| 2 | Motivation / residual degeneration |
| 3 | Algorithm / complexity sketch |
| 4 | **Pareto** (killer) |
| 5 | Budget sweep |
| 6 | Ablation + silence |
| 7 | Behavior + gate stability |
| 8 | Demo / packet-loss (optional) |

---

## Do NOT add before submission
- Causal utility / AC-DSGF++ claims in main
- VQ-VAE, world model, physical topology control
- Retuning λ to chase Success

## Next concrete edits (remaining boxes above)
1. Silence + Pareto caption polish  
2. Limitations soft-packet sentence  
3. Full `pdflatex`+`bibtex` → `AC_DSGF_v1.1.pdf`  
4. Cover letter highlight bullets matching R1–R3
