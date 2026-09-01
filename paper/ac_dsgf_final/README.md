# AC-DSGF Submission Pack (v1)

**PDF:** `main.pdf` (= `../ac_dsgf/AC_DSGF_v1.pdf`) — **7 pages** IEEE draft  
Compiled: 2026-07-15

## Week-1 done
- [x] Full LaTeX compile (`pdflatex` ×3 + `bibtex`)
- [x] Notation table (`notation_table.tex`) — clarifies \(g_{ij}\) is soft \(\in[0,1]\)
- [x] Cover letter draft
- [x] Mock response skeleton (R1/R2/R3)
- [x] Caption language hardened (comparable / graceful degradation)

## Claim (locked)
Maintains task effectiveness under communication constraints; \(\sim\)90× Comm reduction vs DSGF at \(N=16\).

## Week-2 (optional / RA-L)
```bash
python scripts/eval_packet_loss.py --episodes 64 --plot
```
Then add Fig packet-loss + short subsection. Demo 16/32 for presentation only.

## Do not
- Retrain / retune λ / chase Success

## Recompile
```bash
cd paper/ac_dsgf
pdflatex ac_dsgf_main.tex && bibtex ac_dsgf_main && pdflatex ac_dsgf_main.tex && pdflatex ac_dsgf_main.tex
copy ac_dsgf_main.pdf AC_DSGF_v1.pdf
```
