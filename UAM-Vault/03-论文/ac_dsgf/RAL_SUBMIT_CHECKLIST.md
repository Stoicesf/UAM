# RA-L Submit Package — Executable Checklist

**Markdown source of truth:** `AC_DSGF_EN.md` / `AC_DSGF_CN.md`  
**PDF shell:** `ac_dsgf_main.tex` (synced abstract/title/conclusion to freeze)  
**Defense card:** `response_mock.md`

---

## 0) Triple alignment (read aloud once)

| Slot | Sentence |
|------|----------|
| **Title** | Learning Adaptive Topologies for Communication-Constrained UAV Swarm Coordination |
| **Abstract end** | Under identical communication budgets … competitive coordination … substantially lower **SCA** |
| **Contribution 2** | Differentiable adaptation with budgets: Candidate Edge Scoring → Budget-Constrained Edge Selection → Residual Recovery |

If a reader can restate: *topology = soft-budget decision variable; advantage = SCA not brute Success* → narrative is locked.

---

## 1) Main PDF (hard gates)

| Check | Target | How |
|-------|--------|-----|
| Template | IEEEtran **journal**, 10pt (RA-L official) | `ac_dsgf_main.tex` uses `[letterpaper,10pt,journal]` — *not* conference (conference = ICRA/IROS) |
| Body pages | **≤ 6** (+ refs unlimited) | After compile: count pages before bibliography |
| Fonts | Type 1 embedded | Overleaf OK; local: `\pdfminorversion=7` already set |
| Size | ideally **≤ 2 MB** | `gs … /prepress` if needed (see §1b) |
| Figs | ≥300 dpi; consecutive **Fig. 0–11** cited | Index in MD; filenames may keep historical prefixes |
| Table 2 | contains **~2,600×** + hard-budget infeasible note | Already in MD; ensure tex/experiments carries it before final PDF |

### 1b) Compile commands (from `paper/ac_dsgf/`)

```bash
pdflatex -interaction=nonstopmode ac_dsgf_main
bibtex ac_dsgf_main
pdflatex -interaction=nonstopmode ac_dsgf_main
pdflatex -interaction=nonstopmode ac_dsgf_main
```

Compress if needed:

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress -dNOPAUSE -dQUIET -dBATCH -sOutputFile=ac_dsgf_main_submit.pdf ac_dsgf_main.pdf
```

---

## 2) Supplementary zip (lean only)

```
supplementary/
├── appendix.pdf      # arch + hyperparams + Prop.1 derivation (≤2pp)
├── videos/           # optional <60s total
│   ├── baseline_GAT.mp4
│   └── AC_DSGF.mp4
└── README.txt
```

**Forbidden in zip:** new high-Success tables, SwarmOS/PX4 claims, multi-λ curves claimed as main results.

Build appendix:

```bash
cd supplementary
pdflatex -interaction=nonstopmode appendix
```

Zip:

```bash
# after appendix.pdf exists; add videos only if ready
zip -r ../ac_dsgf_supplementary.zip appendix.pdf README.txt videos/
```

---

## 3) Submit-minus-one-minute checklist

- [ ] Main PDF body ≤ 6 pages  
- [ ] Main PDF size checked (< 2 MB preferred)  
- [ ] Fig. 0–11 all cited; no gaps in display numbers  
- [ ] Table 2 has **~2,600×** and hard-budget infeasible wording  
- [ ] Supplementary has **no** new Success experiments  
- [ ] Form checkbox: videos only if `videos/*.mp4` present  
- [ ] `response_mock.md` printed / bookmarked for rebuttal  

---

## 4) P3 (optional, camera-ready)

Do **not** rewrite Fig. 11 (budget–performance operating curve) narrative.  
If λ sweep is run: gray dots + one sentence “coarse λ sweep exhibits similar trend.”

---

## Compile status (2026-07-18)

| Artifact | Result |
|----------|--------|
| `ac_dsgf_main.pdf` | **Built** (~1.4 MB, Type1 fonts via MiKTeX) — currently **8 pages total** (body+figs+refs). **Must compress to ≤6 body pages before Submit.** |
| `supplementary/appendix.pdf` | **Built** (1 page) |
| `ac_dsgf_supplementary.zip` | **Built** (`appendix.pdf` + `README.txt`; no videos yet) |

### Page-cut priority (to hit ≤6 body)
1. Shrink / merge Fig. budget + behavior panels; drop duplicate Pareto if MD Fig.11 already covers it  
2. Move long tables (packet-loss / complexity) to appendix  
3. Tighten related-work to ≤0.5 column  

**Do not** change narrative freeze while cutting pages.

**Note on template:** RA-L official IEEEtran option is **`journal`** (not `conference`). Conference mode is for ICRA/IROS.

| Artifact | Path |
|----------|------|
| EN MD | `paper/ac_dsgf/AC_DSGF_EN.md` |
| CN MD | `paper/ac_dsgf_cn/AC_DSGF_CN.md` |
| Main TeX | `paper/ac_dsgf/ac_dsgf_main.tex` |
| Main PDF | `paper/ac_dsgf/ac_dsgf_main.pdf` (after compile) |
| Appendix TeX | `paper/ac_dsgf/supplementary/appendix.tex` |
| Response mock | `paper/ac_dsgf/response_mock.md` |
| This checklist | `paper/ac_dsgf/RAL_SUBMIT_CHECKLIST.md` |
