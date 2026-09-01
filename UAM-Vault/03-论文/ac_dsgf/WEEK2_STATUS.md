# Week-2 Status

Updated: 2026-07-15 evening

## Done
- [x] Abstract rewrite (comparable + over one order of magnitude)
- [x] Intro core sentence: topology as learnable decision process
- [x] Title → communication-aware / adaptive topology
- [x] Packet-loss experiment (`fig_packet_loss.png`, Table)
- [x] Inference cost Table V (`table5_compute_cost.csv`)
- [x] Complexity note: $K\ll N$ advantage grows with $N$
- [x] Recompiled `AC_DSGF_v1.pdf`

## Packet-loss takeaway (claim-safe)
| Method | p=0 → p=70% |
|--------|-------------|
| GAT | 21.2% → 15.8% (↓ ~25% rel.) |
| AC-DSGF | 26.2% → 25.8% (**flat**) |

Narrative: graceful degradation — not “highest success”.

## Remaining Week-2 (optional polish)
- [ ] Supplement.pdf (hyperparams / architecture)
- [ ] 300dpi audit remaining figures
- [ ] De-AI final pass on Conclusion/Related
- [ ] Cover letter author fill-in
