# T-RO Experimental Design (Frozen)

**Paper:** AC_DSGF_TRO · Phase Experiment Design  
**Status:** **Frozen design** — do not change hypotheses / metrics mid-run without version bump  
**Version:** v0.1  
**Rule:** Experiments **validate theoretical predictions**; they do **not** prove theorems.  
**Out of scope:** strong Theorem 3 (cancelled) · claiming closed-loop \(\rho^\star\approx\rho\) · engineering SwarmOS/PX4 as scientific claims  
**In scope (theory companion):** complexity proposition \(C=O(N)\) under fixed-\(K\) projection

---

## 0. Theory → Experiment map

| Theory | Empirical question | Section |
|--------|--------------------|---------|
| **Thm.~1** | Does \(C(G_t)\le B_t\) always? Does density scale as \(O(1/N)\) under fixed degree \(K\)? | §6.2 |
| **Lem.~2** | Does \(D_G=\|A-A^\star\|_F\) track \(\varepsilon_G=\|M(G)-M(G^\star)\|\)? | §6.3 Fig.~1 |
| **Thm.~2** | Does larger \(\varepsilon_G\) associate with larger shared-state / twin \(\Delta J\)? | §6.3 Fig.~2 |
| Tradeoff | How does budget fraction trade with reward? | §6.3 Fig.~3 |
| Prop. complexity | Is \(C_N=O(N)\) under fixed \(K\)? | §6.4 — **empirical** (Thm.~3 cancelled) |
| Robustness | Does adaptive topology hold under loss/delay? | §6.5 |
| Positioning | Advantage vs CommNet / TarMAC / IC3Net class? | §6.6 |

**Full theory chain under test:**
\[
\phi_\theta(s)
\;\rightarrow\;
G_t=\Pi_{B_t}(S_t)
\;\rightarrow\;
C(G_t)\le B_t
\;\rightarrow\;
\|A-A^\star\|_F
\;\rightarrow\;
\varepsilon_G
\;\rightarrow\;
|J^\star-J|.
\]

---

## 1. Manuscript Section 6 outline (locked)

```
6 Experiments
  6.1 Experimental Setup
  6.2 Validation of Budget-Constrained Topology Projection     [Thm 1]
  6.3 Validation of Topology–Performance Tradeoff              [Lem 2 + Thm 2]
  6.4 Scalability Analysis with Increasing Swarm Size          [FROZEN · empirical + Prop.]
  6.5 Robustness under Realistic Communication Constraints
  6.6 Comparison with Communication Learning Baselines
```

**Forbidden framing:** “We compare AC-DSGF with baselines” as the organizing thesis.  
**Required framing:** theory-driven validation of feasibility and topology-induced performance stability.

---

## 2. Setup (§6.1)

### 2.1 Tasks (UAV swarm aligned)

| ID | Task | Primary metrics |
|----|------|-----------------|
| T1 | Cooperative Navigation | success, collision, return \(J\) |
| T2 | UAV Coverage / Exploration (**strongly recommended**) | coverage ratio, exploration time, \(J\) |
| T3 | Target Tracking | tracking error, success, \(J\) |

Keep T1 for continuity with prior work; T2/T3 differentiate from generic navigation-only MARL.

### 2.2 Backbone / training (implementation freeze)

- Policy stack: frozen AC-DSGF + MAPPO (no new modules for this phase).  
- Topology: \(G_t=\Pi_{B_t}(\phi_\theta(s_t))\); report both soft training and hard projection at eval.  
- Seeds: ≥3 (prefer 5) independent runs; report mean ± std (or CI).

### 2.3 Baseline taxonomy (must cover all three classes)

**A. No communication learning**

| Method | Role |
|--------|------|
| Independent PPO / IPPO | communication necessity (negative control) |
| MAPPO (no / minimal msg) | strong non-comm / centralized-critic baseline |

**B. Fixed-graph communication**

| Method | Role |
|--------|------|
| CommNet | fixed / unstructured pooling |
| GAT-MAPPO | fixed support + attention aggregation |
| DGN | graph MARL on fixed / radius graph |

**C. Learned communication**

| Method | Role |
|--------|------|
| TarMAC | learned whom-to-address |
| IC3Net | learned gating |
| ATOC | attentional communication |
| MAGIC (if implementable) | learned graph communication |

**Claim boundary:** advantage is **joint topology decision + budget projection**, not “we sparsify better than everyone on Success alone.”

### 2.4 Global metrics glossary

| Symbol | Definition |
|--------|------------|
| \(\rho_t\) | communication density \(\lvert E_t\rvert / [N(N-1)]\) (directed; adjust if undirected) |
| \(\bar d\) | mean degree \(N^{-1}\sum_i d_i\) |
| \(V_B\) | budget violation \(\max(0, C(G_t)-B_t)\) |
| \(D_G\) | \(\|A-A^\star\|_F\) |
| \(\varepsilon_G\) | \(\|M(G)-M(G^\star)\|\) (same encoder \(M\), twin state) |
| \(\Delta J\) | \(\lvert J^\star-J\rvert\) under shared-state / twin protocol when possible |
| \(C_N\) | communication cost at size \(N\) (edges / SCA / bits — pick one primary and stick to it) |
| \(SE_N\) | scaling efficiency \(J_N / C_N\) |

---

## 3. Experiment 1 — Budget feasibility (§6.2) · validates Thm.~1

### Hypotheses (pre-registered)

- **H1.1** Under hard projection \(\Pi_{B_t}\), \(V_B=0\) at every eval step (up to numerical ties).  
- **H1.2** Under fixed per-agent degree \(K\), \(\bar d \approx K\).  
- **H1.3** Under fixed \(K\), density \(\rho \sim O(1/N)\) as \(N\) grows.

### Factors

| Factor | Levels |
|--------|--------|
| \(N\) | \(\{8,16,32,64\}\) |
| Degree budget \(K\) | \(\{2,4,8\}\) (primary) |
| Optional density target \(\rho\) | report if using global edge budget instead of degree |

### Protocol

1. Train / load policy with projection enabled at evaluation.  
2. Roll out episodes; log \(C(G_t)\), \(B_t\), \(\lvert E_t\rvert\), degrees.  
3. Aggregate \(\rho\), \(\bar d\), \(V_B\) (max and mean).

### Pass / fail criteria (for writing, not for hacking)

| Check | Pass |
|-------|------|
| \(V_B\) | identically 0 under hard \(\Pi_{B_t}\) |
| \(\bar d\) | within tolerance of \(K\) (e.g. \(\pm 0.05\) if always Top-\(K\)) |
| \(\rho\) vs \(N\) | log-log or \(N\rho\) roughly flat for fixed \(K\) |

### Figures / tables

- Table: \(N\times K\) with \(\rho\), \(\bar d\), \(V_B\).  
- Fig: \(\rho(N)\) for each \(K\) (show \(O(1/N)\) guide).

---

## 4. Experiment 2 — Topology–information–performance (§6.3) · validates Lem.~2 + Thm.~2

**Most important experiment in the paper.**

### Hypotheses

- **H2.1 (Lemma 2 trend):** \(D_G\) and \(\varepsilon_G\) are positively associated (near-monotone / roughly linear under fixed \(M\)).  
- **H2.2 (Theorem 2 trend):** \(\varepsilon_G\) and \(\Delta J\) are positively associated under **shared-state / twin** evaluation when available.  
- **H2.3 (Tradeoff):** Increasing budget fraction improves \(J\) with diminishing returns; not “more edges always better without bound.”

### Factor — budget fraction relative to full information

\[
\frac{B}{B_{\mathrm{full}}}
\in
\{0.1,\,0.2,\,0.4,\,0.6,\,0.8,\,1.0\}
\]
(or equivalent \(K\) ladder that spans sparse → near-full).

### Twin / shared-state protocol (align with Thm.~2)

Preferred:
1. Roll a reference trajectory under \(G^\star=G_{\mathrm{full}}\) **or** fix state sequence from one policy.  
2. On the **same** \(\{s_t\}\), recompute messages/actions under sparse \(G_t\) (twin).  
3. Record \(D_G\), \(\varepsilon_G\), \(\Delta J=\lvert J_T^\star-J_T\rvert\).

If twin recompute is engineering-heavy: report (i) twin proxy on logged states, and (ii) separate closed-loop \(\Delta J_{\mathrm{CL}}\) as **secondary**, clearly labeled as outside Thm.~2’s formal scope.

### Required figures

| Fig | Axes | Theory link |
|-----|------|-------------|
| Fig.~1 | \(D_G\) → \(\varepsilon_G\) | Lemma 2 |
| Fig.~2 | \(\varepsilon_G\) → \(\Delta J\) | Theorem 2 **trend** |
| Fig.~3 | \(B/B_{\mathrm{full}}\) → Reward \(J\) | practical tradeoff |

### Pass / fail (qualitative)

- Fig.~1: clear positive slope; report Spearman / Pearson.  
- Fig.~2: positive association; **do not** claim the numerical constant \(L_R L_\pi/(1-\gamma)\) is identified.  
- Fig.~3: monotone-ish budget–reward curve with variance bars.

---

## 5. Experiment 3 — Scalability study (§6.4) · **FROZEN** (empirical + complexity proposition)

### Hypotheses (observational)

- **H3.1** Under fixed \(K\), \(C_N = O(N)\) (e.g. \(\lvert E\rvert\le NK\)).  
- **H3.2** \(J_N\) remains competitive vs full communication as \(N\) grows.  
- **H3.3** Scaling efficiency \(SE_N=J_N/C_N\) favors projected topology over full connectivity.

### Factors

| Factor | Levels |
|--------|--------|
| \(N\) | \(16,32,64,128\) (+ \(256\) if compute allows) |
| Methods | AC-DSGF (projected) vs Fully-connected communication |

### Metrics

\(J_N\), \(C_N\), \(SE_N\); optional degree histogram stability.

### Theorem 3 trigger (**CANCELLED** — keep empirical + complexity proposition)

1. Empirically \(C_N=O(N)\).  
2. \(J_N\) stable / gracefully degrading vs full graph.  
3. Topology statistics (degree / \(D_G\)) show regular structure across \(N\).

**Decision (2026-07-24):** keep §6.4 empirical; write complexity proposition only; do not draft Thm.~3.

---

## 6. Experiment 4 — Realistic channel (§6.5)

### Attack neutralized

> “Communication cost is only an abstract count.”

### Factors

| Channel | Levels |
|---------|--------|
| Packet loss \(p_\ell\) | \(\{0, 0.1, 0.3, 0.5\}\) |
| Delay \(\tau\) | \(\{0, 20, 50, 100\}\) ms (sim-step equivalent OK) |
| Bandwidth / \(B_t\) | match §6.2–6.3 budgets |

### Metrics

task reward, collision, success rate, **topology adaptation** (how \(\rho\) / degree shifts under impairment).

### Hypotheses

- **H4.1** Fixed dense / fixed graph methods degrade sharply under loss/delay.  
- **H4.2** Budgeted adaptive topology retains higher \(J\) / success at matched or lower \(C\).

---

## 7. Experiment 5 — Baseline comparison (§6.6)

### Protocol

- Same tasks (at least T1 + T2), matched compute / message dimension where fair.  
- Report **joint** \((J, C)\) or \((J, \rho)\), not Success alone.  
- Identical-budget slice when comparing learned communicators.

### Narrative

1. Class A → communication helps.  
2. Class B → dynamic / projected topology beats fixed support at same budget.  
3. Class C → topology-as-decision + \(\Pi_{B_t}\) differs from message-content learning on fixed supports.

---

## 8. Reporting rules (claim hygiene)

| Allowed | Forbidden |
|---------|-----------|
| “Consistent with Thm.~1 / Lem.~2 / Thm.~2” | “Experiments prove Theorem 2” |
| “Positive association \(\varepsilon_G\)–\(\Delta J\)” | “We identify \(L_R,L_\pi\)” |
| Twin \(\Delta J\) labeled shared-state | Silent closed-loop as if Thm.~2 |
| \(V_B=0\) under hard projection | Soft training alone “enforces” hard \(B\) |
| Observe \(C_N=O(N)\) | Premature Thm.~3 from one plot |

---

## 9. Implementation checklist (before first run)

- [ ] Logging: adjacency \(A_t\), cost \(C\), messages \(m_t\), twin hooks for \(M(G^\star)\)  
- [ ] Hard \(\Pi_{B_t}\) flag at eval  
- [ ] Config grid for \(N,K\), budget fractions, channel params  
- [ ] Baseline env wrappers for Class A/B/C  
- [ ] Figure scripts for Fig.~1–3 locked to axis definitions above  
- [ ] No paper-text edits that invent results before tables exist  

---

## 10. Related stubs / implementation layer

| File | Role |
|------|------|
| [`tro_logging_protocol.md`](tro_logging_protocol.md) | **Frozen** — theory variables → log fields → figures |
| [`tro_evaluation_protocol.md`](tro_evaluation_protocol.md) | **Frozen** — E0/E1/E2 modes, phase order, G1/G2 baselines |
| [`tro_reproducibility_checklist.md`](tro_reproducibility_checklist.md) | Gate before writing §6 results |
| [`channel_model.md`](channel_model.md) | §6.5 parameter notes |
| [`coverage_task.md`](coverage_task.md) | Task T2 details |
| [`baselines.md`](baselines.md) | Class A/B/C notes |

---

## 11. Next action after freeze

1. Implement logging + twin evaluation **hooks** (no model tuning).  
2. Run §6.2 (Thm.~1) first — cheapest, hardest fail if broken.  
3. Run §6.3 (Lem.~2 + Thm.~2) — core figures.  
4. Then §6.4–6.6.  
5. §6.4 frozen empirically; complexity proposition written; Thm.~3 cancelled.
