# Self-Evolving Constrained Distributed Optimization (SECDO)

**Venue target:** T-RO / TPAMI / JMLR / NeurIPS / ICML class  
**Status:** **SUBMIT** — SECDO-v2.0-submission（P0 defense patches frozen）  
**Version:** SECDO-v2.0-submission · PDF: `main.pdf` · Cover: [`COVER_LETTER.md`](COVER_LETTER.md) · ZH: [`SECDO_PAPER_ZH.md`](SECDO_PAPER_ZH.md)  
**LaTeX:** `main.tex` + `sections/` + `appendix/` + `figures/`  
**Claim:** predictive constraint-evolution · PI-conditioned feasibility · failure-safe \(\alpha\)（非 Pareto / 非 enhanced PGD / 非 objective superiority）  
**Baseline (frozen, cited not rewritten):** AC-DSGF · T-RO Track B v0.11（勿改 `paper/ac_dsgf_tro/`）  

> **Naming lock.** Do **not** call this track “AC-DSGF++” in code or configs — that name is already used by Causal-Utility `ac_dsgf_pp`.  
> Paper working titles (pick one later):
> - *Self-Evolving Constrained Distributed Optimization*
> - *Learning the Dynamics of Objectives and Feasible Regions*
> - Informal shorthand: **SECDO** / **LDOFR**

---

## Elevator claim (top-tier)

\[
\boxed{
\textbf{Self-Evolving Constrained Distributed Optimization:}
\text{ jointly learn environment evolution and reconstruct future feasible regions}
}
\]

**Scientific question (one level above “prediction plugins”):**

> In unknown, nonstationary, dynamically constrained environments, can a distributed optimizer **simultaneously learn environment evolution** and **actively reconstruct its future feasible set**?

**Not sufficient for this track:**
> “AC-DSGF with a predictive module / world-model head.”

---

## Three breakthroughs (must all appear)

| # | Breakthrough | One-line |
|---|--------------|----------|
| 1 | **Learnable constraint dynamics** | \(B_{t+1}=\mathcal{F}_\phi(B_t,s_t)\) — constraints are a dynamical system, not exogenous inputs |
| 2 | **Prediction-aware optimization theory** | Regret / gap bounds that **explicitly contain** \(\sum\epsilon_t\) (and \(\delta\)) |
| 3 | **Anticipatory projection** | \(x_{t+1}=\Pi_{\hat B_{t+k}}(y_t)\) — enter future feasible regions, not only correct past violations |

---

## Relation to C1 (Predictive AC-DSGF)

| | C1 (fast / strong conference) | SECDO (top-tier) |
|--|-------------------------------|------------------|
| Object | Predict \(\hat s,\hat B\) to help AC-DSGF | Make **constraint dynamics + anticipatory projection** first-class |
| Theory | Optional feasibility patch | **Required** prediction-aware convergence |
| Reviewer risk | “Just a predictive module” | Must prove paradigm, not plug-in |
| Role of C1 | **Implementation scaffold / ablation** | Subsystem, not the paper title |

C1 engineering (latent dynamics + utility/budget heads) may still be built as the **first executable instantiation**, but the **paper contribution is SECDO**, not the heads.

---

## Locked decisions (Phase 1a)

\[
\boxed{
\begin{aligned}
&\textbf{Thm2: Optimization gap (main)}\\
&\textbf{Regret: Corollary}\\
&\textbf{E3: UAV dynamic bandwidth allocation}\\
&\textbf{Core novelty: anticipatory projection + }\\
&\quad\textbf{learned constraint evolution }\mathcal{F}_\phi
\end{aligned}
}
\]

**Formal freeze:** [`phase1a_formal_spec.md`](phase1a_formal_spec.md)  
**UAV \(c_t\) teacher:** [`phase1b_uav_c_teacher.md`](phase1b_uav_c_teacher.md)  
**Thm1 (formal):** [`theory/thm1_scalar_budget_stability.md`](theory/thm1_scalar_budget_stability.md)  
**Lemma2 (formal, conditional):** [`theory/anticipatory_projection_lemma.md`](theory/anticipatory_projection_lemma.md)  
**Thm2 (formal, gap not “convergence”):** [`theory/thm2_prediction_aware_gap.md`](theory/thm2_prediction_aware_gap.md)  
**Algorithm 1:** [`algorithm_secdo.md`](algorithm_secdo.md)  
**Phase 2 alignment:** [`phase2_alignment.md`](phase2_alignment.md)  
**Code:** `models/secdo/` · `experiments/secdo_uav/` · `scripts/run_secdo_phase2.py`

## Directory

| Path | Role |
|------|------|
| [`phase1a_formal_spec.md`](phase1a_formal_spec.md) | **FROZEN** symbols \(\mathcal{F}_\phi,\hat c,\Pi_{\hat B}\) |
| [`phase0_scientific_problem.md`](phase0_scientific_problem.md) | Elevated problem |
| [`formulation_draft.md`](formulation_draft.md) | Constraint dynamics + anticipatory operator |
| [`theory_targets.md`](theory_targets.md) | Thm 1–3 + corollary |
| [`architecture.md`](architecture.md) | Top-tier system diagram |
| [`experiments_plan.md`](experiments_plan.md) | E1–E4 (E3 = UAV bandwidth) |
| [`roadmap.md`](roadmap.md) | Phase status |
| [`hooks_from_codebase.md`](hooks_from_codebase.md) | Code hooks |
| [`naming.md`](naming.md) | Collision with `ac_dsgf_pp` |

---

## Hard discipline

- Do **not** edit `paper/ac_dsgf_tro/` scientific claims for this track.  
- Do **not** ship Transformer-only aggregators as the main contribution.  
- Do **not** revive cancelled T-RO Thm.~3 performance generalization as the flagship result.  
- Theory without \(\epsilon\)/constraint-dynamics in the statement = not top-tier ready.
