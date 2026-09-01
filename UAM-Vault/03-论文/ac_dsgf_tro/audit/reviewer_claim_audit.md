# Reviewer Claim Audit — AC-DSGF-TRO v0.10 → v0.11

**Scope:** EN/CN main text + theory notes + experiment protocols (wording only).  
**Rule:** No new modules / experiments / theorems. Thm.~3 stays cancelled. RA-L untouched.

---

## High-priority findings (applied in v0.11)

| 位置 | 原句 / 原标题 | 风险 | 修改建议 (v0.11) |
|------|---------------|------|------------------|
| Title | “… for **Scalable** Multi-Agent Systems” | overclaim “scalable” as product property | Soften → “… under **Resource Constraints**” |
| §1.3 | 3 contributions mixing theory+algo | structure weak for reviewers | Expand to **4** crisp contributions |
| Abstract | long; “Prove” vibe via Thm list | dense / slight over-sell | Keep core formulation; shorten empirics |
| §6 intro | “theory-driven **validation**” | implies theorems validated | → “theory-**aligned** empirical examination” |
| §6.5 title | “**Robustness** under …” | “robust guarantee” reading | → “**Performance** under Imperfect …” |
| §6.4 quote | “**substantially** higher … efficiency” | mild superlative | → “**markedly higher** … (empirical)” |
| §6.5 quote | already softened | OK | keep “more stable … under degraded …” |
| Contrib. | “**Prove** feasibility…” | OK if Thm.~1 exact; tone | Prefer “**Establish** …” |
| CN title | “可扩展多智能体…” | same as EN | → “资源约束下…” |
| CN §6.5 | “鲁棒性” | same | → “不完美信道下的性能表现” |

---

## Keyword sweep (EN main text)

| Keyword | Hits / context | Verdict |
|---------|----------------|---------|
| guarantee / guaranteed | Thm.~1 hard feasibility “guaranteed whenever \(\Pi_{B_t}\) applied” | **OK** (constraint feasibility, not task performance) |
| prove / proves | “do not prove lemmas/theorems”; Thm.~2 “does not prove closed-loop…” | **OK** |
| optimal / optimality | always qualified as *utility-maximizing projection*, not task optimality | **OK**; wording locks present |
| generalize / generalization | only in negations (“not … generalization”) | **OK** |
| scalable | title + §6.4 “empirical scalability analysis” | Title **fixed**; section noun OK if empirical |
| robust / robustness | §6.5 title / Fig.~6 filename | Title **softened**; filename may keep (artifact) |
| always | not used as absolute performance claim in EN body | **OK** |
| superior / outperform | absent in EN body; protocol forbids | **OK** |
| novel | absent | **OK** |
| validate | “validation of …” in §6 organizing sentence | **Fixed** → aligned examination |

---

## Residual risks (accepted / documented)

| Item | Note |
|------|------|
| Fig.~6 filename `Fig6_channel_robustness.png` | Keep artifact name; caption/title use “performance under imperfect channels” |
| “utility-maximizing” | Correct under linear score; must stay next to “not task-return optimality” |
| Class C deferred | Mock Reviewer 2 answer ready; do not invent numbers |
| Title still mentions coordination | Fine; core claim intact |

---

## Forbidden → allowed (checklist applied)

| Forbidden | Allowed replacement |
|-----------|---------------------|
| guarantees performance improvement | provides bounded shared-state degradation (Thm.~2) / empirically observed tradeoffs |
| scalable to large swarms | exhibits linear communication growth under fixed-degree projection |
| robust communication | maintains stable performance under degraded communication conditions |
| validates theorem | consistent with |
| optimal topology | budget-feasible utility-maximizing projection / full-support **reference** \(G^\star\) |
