# AC-DSGF-TRO Submission Package Checklist

```text
AC_DSGF_TRO Submission Package

Paper
 ├── Main manuscript (EN/CN)   v0.11 Submission Candidate
 ├── Supplementary              pending final pack (theory/*.md)
 ├── BibTeX                     references/ac_dsgf_tro.bib  ✅
 ├── Figures 1–8                frozen ✅
 └── Tables I–III              frozen ✅

Theory
 ├── Thm1                       frozen ✅
 ├── Lemma1/2                   frozen ✅
 ├── Thm2 (shared-state)        frozen ✅
 └── Prop Complexity           frozen ✅

Experiments
 ├── Budget feasibility §6.2    frozen ✅
 ├── Theory-aligned eval §6.3   frozen ✅
 ├── Scalability §6.4           frozen ✅
 ├── Channel eval §6.5          frozen ✅
 └── Baselines §6.6A           frozen ✅  (Class C deferred)

Audits
 ├── reference_audit.md         ✅
 ├── latex_submission_check.md  ✅
 ├── v0.11_final_reading.md     ✅
 └── final_reviewer_simulation.md ✅
```

- [x] Supplementary packaged (`supp/`)
- [x] Submission gate drafted (`audit/submission_gate.md`)
- [ ] T-RO LaTeX template migration (`audit/latex_migration_notes.md`)
- [ ] PDF compiled + visual inspection

## Path to submit

```text
BibTeX audit ✅
      ↓
LaTeX formatting checklist ✅
      ↓
Supplementary packaging ✅
      ↓
T-RO LaTeX migration   ← next
      ↓
Compile + visual audit
      ↓
submission_gate.md → Submit
```

## Discipline (locked)

- No new theory  
- No claim strengthening  
- No Class C unless reviewer gap appears  
- No RA-L edits  
