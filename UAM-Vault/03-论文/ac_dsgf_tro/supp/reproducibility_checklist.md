# S5 — Reproducibility Checklist (Submission Package)

- [x] Code / config structure documented (`configs/ac_dsgf/`, `scripts/collect_tro_*.py`)
- [x] Random seeds reported: \(\{1234,2026,3407,42,8888\}\)
- [x] Communication cost \(C\) definition method-aware (§6.6 / S3.3)
- [x] Budget constraints specified (\(B_t\), fixed-\(K\), ratio budgets)
- [x] Theoretical assumptions listed (Thm.~1 feasible set; Lem.~2 Assump. M; Thm.~2 A1–A3; shared-state only)
- [x] Evaluation protocols separated (E0 closed-loop vs E1 twin / shared-state)
- [x] Checkpoint roots documented (no silent retrain for formal tables)
- [x] Claim hygiene: no Thm.~3; no closed-loop guarantee; Class C deferred
- [x] Supplementary linked from package README
- [ ] Camera-ready: compile under IEEE T-RO LaTeX template (pending migration)
- [ ] Camera-ready: BibTeX `\cite{}` keys wired from `references/ac_dsgf_tro.bib` (pending migration)

Parent experimental checklist: [`../experiments/tro_reproducibility_checklist.md`](../experiments/tro_reproducibility_checklist.md).
