# AC-DSGF Submission Checklist — Submission Hardening
# Updated after RA-L Final Submission Checklist

> Algorithm frozen. Only writing / packaging. **No ++.**

## Day 1 — Trio (DONE)
- [x] Cover Letter final (`paper/ac_dsgf/COVER_LETTER.md`)
- [x] Highlights H1–H4 (`paper/ac_dsgf/HIGHLIGHTS.md`)
- [x] Contribution Statement ×3 (`paper/ac_dsgf/CONTRIBUTION_STATEMENT.md`)

## Day 2 — Runtime + figures
- [x] Runtime/scalability table in `complexity.tex` + `paper/tables/table_runtime_scalability.csv`
- [ ] Export figures ≥300 dpi into `AC_DSGF_RA-L_Submission/figures/`
- [ ] Recompile main PDF after complexity.tex change

## Day 3 — Supplement + pack
- [x] S5 as *Additional Exploration: Outcome-Aware…* (not Failure Analysis)
- [x] Folder `paper/AC_DSGF_RA-L_Submission/` + `README.txt`
- [ ] Export Cover / Highlights / Supplement to PDF
- [ ] Zip for ScholarOne

## Day 4 — Submit RA-L
- [ ] ScholarOne upload
- [ ] Keep video optional

## PDF hard checks
- [x] Abstract opens: why communication optimization
- [x] Abstract closes: effectiveness under communication constraints
- [x] Table I: *Task performance under communication constraints*
- [x] Comm = soft mass \(C=\frac1T\sum_t\sum_{ij}g_{ij}^{t}\)
- [x] Do not lead with AC Success > DSGF
- [x] Lead CEI / Comm under Success-hold
- [ ] Figure order: Mot+FW → Method → Trade-off → Budget → Ablation → Behavior

## Frozen evidence (do not regenerate unless broken)
| Item | Path |
|------|------|
| Table I | `paper/tables/table1_final.csv` |
| Compute | `paper/tables/table5_compute_cost.csv` |
| Runtime scale | `paper/tables/table_runtime_scalability.csv` |
| Demo | `demo/videos/ac_dsgf_dynamic_comm.mp4` |

## After RA-L
Decide **AC-DSGF 2.0: Action–Communication Co-evolution** from reviews — not from ++ revival.
