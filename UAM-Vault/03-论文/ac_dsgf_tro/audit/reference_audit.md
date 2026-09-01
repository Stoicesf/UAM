# Reference Audit — v0.11 Package Completion

**Goal:** Positioning clarity in three axes — not reference count.  
**BibTeX:** [`../references/ac_dsgf_tro.bib`](../references/ac_dsgf_tro.bib)  
**Rule:** 2–4 cites per claim; no 10-key dumps.

---

## 1.1 Category coverage

| 类别 | 目的 | 检查 | Core keys (≤) |
|------|------|------|----------------|
| Comm MARL | 已有工作偏 message / attention | ✅ 经典+近年 | Foerster2016, Sukhbaatar2016, Das2019, Jiang2018, Singh2019 |
| Graph MARL | 图结构协调 / 聚合背景 | ✅ | Velickovic2018, Jiang2020, Kipf2017 |
| Constrained RL | 支撑 \(C(G)\le B\) | ✅ | Altman1999, Achiam2017, Tessler2019 |
| Coordination | 多智能体协作背景 | ✅ | Yu2022, Lowe2017, Bettini2022 |
| Topology adjacent | 区分 structure learning / pruning | ✅ sparingly | Kipf2018NRI, Franceschi2019 (optional) |

**Three-axis map for reviewers**

```text
Communication MARL ───────── message/attention on fixed support
Graph-based Coordination ─── aggregators on G (G rarely decided)
Resource-Constrained DM ──── hard budgets / CMDP / CPO
              ↓
         This work: G_t = φ_θ(s_t) with Π_{B_t}
```

---

## 1.2 Claim → citation map

| Claim (Intro / Related / Discussion) | Needed axis | Citations in manuscript |
|--------------------------------------|-------------|-------------------------|
| Existing communication MARL mainly optimizes message content or aggregation | Comm MARL | §2 names CommNet, TarMAC, ATOC, IC3Net (+ Foerster/DIAL lineage) |
| Graph methods provide relational / local coordination representations | Graph MARL | §2 GAT/DGN (+ Kipf GCN in Refs) — **avoid “scalable” wording** |
| Communication resources introduce explicit constraints | Constrained RL | §2 CMDP / CPO / RCPO (Refs 9–11) |
| \(G_t=\phi_\theta(s_t)\) is a decision, not post-hoc pruning | Topology adjacent + Distinction | §1.2 table + §2 Distinction + §7 |
| Empirics use cost-regime baselines | Coordination | MAPPO, VMAS cited |

**Verdict:** All three positioning claims have 2–4 named anchors. No stacking required.

---

## 1.3 Anti-stacking policy (locked)

| Do | Don't |
|----|-------|
| One paragraph → 2–4 named methods | `\cite{a,b,c,d,e,f,g,h,i,j}` |
| Point to `references/*.md` for expansion | Dump optional surveys into Related Work |
| Keep TarMAC/IC3Net/ATOC as *paradigm family* | Lead with “we did not implement Class C” |

---

## Gaps (non-blocking)

| Gap | Action |
|-----|--------|
| Markdown refs lack `\cite{}` keys | OK until LaTeX port; `.bib` ready |
| DIAL not in `.bib` | Optional; Foerster2016 covers lineage |
| Event-triggered comm papers | Optional under constrained axis if camera-ready expands |

**Class C:** deferred — does not change reference strategy.
