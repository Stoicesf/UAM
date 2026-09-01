# Learning Communication Topology as a Coordination Variable under Resource Constraints

**Subtitle:** A Constraint-Aware Dynamic Graph Decision Framework  
**Method (implementation reuse):** AC-DSGF backbone (frozen; no new modules)  
**Track:** T-RO / TNNLS (B) · **v0.11** · *Submission Candidate · scientific freeze*  
**Center object:** \(\boxed{G_t=\phi_\theta(s_t)}\)  
**Supplementary:** [`supp/AC_DSGF_TRO_supp_EN.md`](supp/AC_DSGF_TRO_supp_EN.md) · **BibTeX:** [`references/ac_dsgf_tro.bib`](references/ac_dsgf_tro.bib)

> This manuscript is **independent** of the RA-L freeze.  
> Do not merge narrative claims from the RA-L SCA-sparsification paper without re-deriving them in the DGDP framing.

**Symbol freeze (entire paper).**

| Symbol | Role | Forbidden |
|--------|------|-----------|
| \(\phi_\theta\) | **topology policy** | writing \(\pi_\theta\) for the graph learner |
| \(\pi_\psi\) | **physical action policy** | conflating with topology scoring |
| \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\) | joint coordination policy | ambiguous \(\Pi=(\pi,\phi)\) without params |
| \(\Pi_{B_t}\) | budget **projection operator** | calling \(\Pi_{B_t}(S_t)\) “the optimal topology” |
| \(G_t^\star\) | full-support **reference** topology before projection | “optimal / best / argmax topology” |

---

## Abstract

Multi-agent coordination policies typically treat the communication graph as fixed infrastructure—radius neighborhoods, \(k\)-nearest neighbors, or all-to-all links—while learning only physical actions. Under resource constraints, we **formulate communication topology as a learnable coordination decision variable jointly optimized with physical policies**. We formalize a *Dynamic Graph Decision Process* (DGDP) whose joint policy \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\) outputs
\[
G_t=\phi_\theta(s_t),\qquad
a_t=\pi_\psi\bigl(s_t,\,M(\phi_\theta(s_t))\bigr),
\]
so that the graph lies in the decision space, not merely in the feature pipeline. Feasible topologies are obtained by an edge-utility estimator followed by a **budget projection operator** \(\Pi_{B_t}\). We establish budget feasibility and utility-maximizing projection under learned edge scores (Theorem 1)—not task-return optimality—together with shared-state information / action / return discrepancy bounds (Lemmas 1–2, Theorem 2) and a complexity proposition for fixed-degree projection. Empirically, multi-seed twin evaluations are consistent with Lemmas 1–2; swarm-size studies exhibit linear communication growth under fixed \(K\); channel stress tests and method-aware \((J,C)\) comparisons complete the program. Overall, the approach achieves **communication-efficient coordination under constrained communication budgets**, instantiated on a frozen AC-DSGF backbone.

**Keywords:** dynamic graph decision process; learnable communication topology; budget projection; multi-agent coordination; resource constraints

---

## 1 Introduction

### 1.1 Topology as infrastructure vs.\ topology as decision

In large multi-agent and multi-UAV systems, communication is limited by bandwidth, energy, and interference. A common modeling choice is to treat the communication graph \(G_t\) as **infrastructure**:
\[
A_t = f_{\mathrm{geo}}(x_t)
\]
(fully connected, radius, or \(k\)-NN), then learn only \(a_t=\pi(s_t)\). This separates coordination structure from decision-making.

We instead place the graph in the **action / decision space**:
\[
\boxed{G_t = \phi_\theta(s_t)},
\qquad
a_t = \pi_\psi\bigl(s_t,\,M(G_t)\bigr).
\]
The joint object is \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\). The operational question is not only *what to do*, but *with whom to communicate under a (possibly time-varying) budget*.

**Primary claim (do not weaken or strengthen).**  
We formulate communication topology as a learnable coordination decision variable jointly optimized with physical policies.  
We do **not** claim to “optimize the communication topology” in the sense of \(\max_G J(\pi,G)\); learned scores enter a projection that maximizes \(\sum g_{ij}\) over \(\mathcal{G}_{B_t}\).

### 1.2 Gap relative to communication / graph MARL

Existing communication MARL methods mainly optimize *message content*, *gating*, or *attention aggregation* (e.g., CommNet, TarMAC, ATOC, IC3Net), typically on fixed or heuristically restricted supports. Graph-based methods (e.g., GAT, DGN) provide strong local / relational coordination representations, but usually treat \(G\) as given infrastructure. Separately, bandwidth, energy, and interference make communication an **explicit resource constraint**—closer to constrained decision making than to unconstrained message learning.

| Line | Typical object | Topology role |
|------|----------------|---------------|
| Message learning (CommNet, TarMAC, …) | message content / attention | support graph largely fixed |
| Graph MARL (GAT, DGN, …) | aggregation on \(G\) | \(G\) default radius/KNN |
| Post-hoc sparsification | prune fixed \(G\) | weak joint task coupling |
| **This work** | \(G_t=\phi_\theta(s_t)\) as decision | budgeted projection + joint \(\Pi_{\theta,\psi}\) |

### 1.3 Contributions

1. **Topology as decision.** We formulate communication topology as a learnable coordination variable jointly optimized with physical policies under explicit resource constraints (DGDP with \(\Pi_{\theta,\psi}\)).  
2. **Theory.** We establish feasibility of budget-constrained topology projection (Thm.~1) and shared-state stability bounds connecting topology approximation, information discrepancy, and action deviation (Lem.~1–2, Thm.~2).  
3. **Communication complexity.** We characterize communication complexity under fixed-degree topology projection: \(C=O(N)\) when \(K=O(1)\) (complexity proposition)—not a graph-size performance theorem.  
4. **Experiments.** We report communication–performance tradeoffs under constrained and imperfect communication: budget feasibility, twin theory evidence, empirical linear communication growth, channel stress tests, and method-aware \((J,C)\) baselines.

**Primary one-line claim.** Budget-constrained topology learning with **linear communication growth under fixed-degree projection.**  
*(Forbidden contribution language: scalable general intelligence; optimal coordination; universal topology learning.)*

---

## 2 Related Work

**Communication learning in MARL.** CommNet, DIAL, TarMAC, ATOC, and IC3Net learn *when/what* to communicate—or *whom* to attend to—typically on fixed or heuristically restricted supports. Explicit membership of the communication graph in a hard feasible set \(\mathcal{G}_{B_t}\) is secondary. (Expanded notes: [`references/communication_marl.md`](references/communication_marl.md).)

**Graph MARL.** GAT/DGN-style methods learn aggregators *on* a graph and provide useful local / relational inductive biases; the graph itself is rarely a constrained decision variable jointly optimized with control ([`references/graph_marl.md`](references/graph_marl.md)).

**Resource-constrained and communication-efficient RL.** Constrained MDPs / constrained policy optimization motivate hard resource limits; event-triggered and post-hoc pruning reduce cost on a *given* graph, but do not define a projection onto \(\mathcal{G}_{B_t}\) ([`references/constrained_rl.md`](references/constrained_rl.md)).

**Coordination setting.** Cooperative MARL and multi-robot simulators (e.g., MAPPO, VMAS) learn physical actions under an assumed interaction structure ([`references/multi_agent_coordination.md`](references/multi_agent_coordination.md)). Adjacent structure-learning / sparsification lines infer or prune graphs without our DGDP + hard \(\Pi_{B_t}\) framing ([`references/related_topology_learning.md`](references/related_topology_learning.md)).

**Distinction.** We elevate \(G_t=\phi_\theta(s_t)\) to a **learnable coordination decision** under \(C(G_t)\le B_t\), obtain feasibility via \(\Pi_{B_t}\), and analyze topology-induced information / action discrepancy—without claiming that \(\Pi_{B_t}\) maximizes task return. Empirically we compare **representative communication-cost regimes** (no communication, radius-sparse, dense, budgeted projection), not an exhaustive attention-mechanism bake-off.

---

## 3 Dynamic Graph Decision Process

### 3.1 Definition 1 (DGDP)

A **Dynamic Graph Decision Process** is
\[
\mathcal{M}_G
=
\bigl(\mathcal{S},\mathcal{A},\mathcal{G},P,R,C,\gamma\bigr),
\]
where \(\mathcal{S}\) is the swarm state space, \(\mathcal{A}\) the physical action space, \(\mathcal{G}\) the universe of communication graphs on agent set \(V\), \(P\) the transition kernel, \(R\) the task reward, \(C\) the communication cost, and \(\gamma\in(0,1]\) the discount.

**Decision structure.** At time \(t\), the topology policy and physical policy act jointly:
\[
G_t=\phi_\theta(s_t),\qquad
m_t=M(G_t)=M\bigl(\phi_\theta(s_t)\bigr),\qquad
a_t=\pi_\psi(s_t,m_t).
\]
**Topology–action coupling (compact form):**
\[
a_t
=
\pi_\psi\bigl(s_t,\,\phi_\theta(s_t)\bigr)
\quad\text{(through message map \(M\))},
\qquad
\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta).
\]
Thus \(G_t\) is part of the **decision**, not only an input feature of \(\pi_\psi\).

*(Decentralized instantiation may replace \(s_t\) by local observations \(o_t\); the coupling structure is unchanged.)*

**Contrast with standard MARL.** Standard MARL outputs \(a_t=\pi(s_t)\) on a fixed communication support. DGDP outputs a **two-layer decision**: topology first, then action conditioned on the induced messages.

```
state s_t
   │
   ├─ topology policy φ_θ  →  G_t = φ_θ(s_t)
   │                            │
   │                            └─ M(·) → m_t
   │
   └─ physical policy π_ψ(s_t, m_t) → a_t
```

### 3.2 Budgets: static and dynamic

Communication cost must satisfy
\[
C(G_t)\le B_t.
\]
- **Dynamic budget** \(B_t\): time-varying capacity (bandwidth fluctuation, interference, energy).  
- **Static budget** \(B\): the special case \(B_t\equiv B\).

Feasible families:
\[
\mathcal{G}_{B_t}=\{G\in\mathcal{G}:C(G)\le B_t\}.
\]

### 3.3 Joint topology–policy optimization

Let \(c(G_t)\) be a training surrogate of communication cost (edge count, soft activation mass, or degree). The objective is
\[
\max_{\theta,\psi}
\;
J(\Pi_{\theta,\psi})
=
\mathbb{E}
\Bigg[
\sum_{t=0}^{T}
\gamma^t
\bigl(r_t-\lambda\, c(G_t)\bigr)
\Bigg]
\]
subject to
\[
G_t=\phi_\theta(s_t),
\qquad
C(G_t)\le B_t.
\]
In boxed form:
\[
\boxed{
\max_{\theta,\psi}
J(\Pi_{\theta,\psi})
\quad
\mathrm{s.t.}
\quad
G_t=\phi_\theta(s_t),\;
C(G_t)\le B_t.
}
\]

**Soft surrogate (training).** Differentiable gates \(g_{ij}=\sigma(s_{ij})\) with penalty \(\lambda\) realize a Lagrangian relaxation; hard feasibility is enforced by the projection \(\Pi_{B_t}\) at decision / deployment time (Sec.~4). Soft training alone is **not** claimed to enforce hard \(B_t\).

### 3.4 Instantiation note (backbone freeze)

Edge scores and residual-corrected actors may be realized by the existing AC-DSGF encoder–actor stack. **No new modules** are introduced in v0.2; the contribution is the DGDP formulation, symbol freeze, and the projection theorem.

---

## 4 Constraint-Aware Topology Projection

### 4.1 Edge utility learner

Given state \(s_t\) (and node embeddings \(h_i\)), the **topology policy** \(\phi_\theta\) produces a score matrix
\[
S_t=\{s_{ij}\}_{i\neq j}
=
\phi_\theta(s_t).
\]
Interpret \(s_{ij}\) as a *learned surrogate* of task-relevant edge utility under the current swarm context—not as ground-truth \(\Delta R_{ij}\) and not as a claim that \(\arg\max_G\sum s_{ij}\) maximizes \(J\).

Continuous gates (training):
\[
g_{ij}=\sigma(s_{ij})\cdot A_{t,ij},
\]
where \(A_t\) is the candidate / geometric support at time \(t\) (e.g., communication radius).

### 4.2 Projection operator \(\Pi_{B_t}\)

Define the **budget projection** (operator abstraction; no commitment to a specific solver):
\[
\Pi_{B_t}(S)
=
\arg\max_{x}
\sum_{ij} g_{ij}\,x_{ij}
\quad
\mathrm{s.t.}
\quad
\sum_{ij}x_{ij}\le B_t,
\quad
x_{ij}\in\{0,1\},
\quad
x_{ij}\le A_{t,ij}.
\]
When the budget is a per-agent degree constraint \(|E_t(i)|\le K\), \(\Pi_{B_t}\) *may* specialize to independent Top-\(K\) on each row of \(S_t\) (as in the frozen AC-DSGF budget layer). Other realizations of \(\Pi_{B_t}\) are allowed so long as they implement the same constrained argmax.

The decided graph is
\[
G_t=\Pi_{B_t}(S_t)=\Pi_{B_t}\bigl(\phi_\theta(s_t)\bigr).
\]

**Wording lock.** Call \(G_t\) a **budget-feasible utility-maximizing projection** under learned edge utilities. Do **not** call it an “optimal topology” or “task-optimal communication graph.”

### 4.3 Theorem 1 — Feasibility of Budget-Constrained Dynamic Topology Projection

**Theorem 1 (Feasibility of budget-constrained dynamic topology projection).**  
Fix candidate support \(A_t\) and budget \(B_t\ge 0\). For any score matrix \(S_t\) (equivalently gates \(g\)), let \(G_t=\Pi_{B_t}(S_t)\) with edge indicators \(x^\star\). Then:

**(1) Feasibility.**  
\[
C(G_t)=\sum_{ij}x^\star_{ij}\le B_t.
\]
If the budget is degree-wise \(|E_t(i)|\le K\), then \(|E_t|\le NK\).

**(2) Utility-maximizing projection (not task optimality).**  
For every feasible \(G'\in\mathcal{G}_{B_t}\) with indicators \(x'\),
\[
\sum_{ij}g_{ij}\,x^\star_{ij}
\;\ge\;
\sum_{ij}g_{ij}\,x'_{ij}.
\]
That is, \(\Pi_{B_t}\) yields a **budget-feasible utility-maximizing projection** under the learned scores \(\{g_{ij}\}\). It does **not** assert
\[
G_t\in\arg\max_{G\in\mathcal{G}_{B_t}} J(\pi_\psi,G).
\]

**Proof sketch.**  
(1) holds by construction of the constraint in \(\Pi_{B_t}\).  
(2) is the definition of the argmax over \(\mathcal{G}_{B_t}\). For separable per-agent Top-\(K\), each row independently retains the \(K\) largest admissible scores, which is optimal for linear objectives separable across agents under degree caps.  
□

**Interpretation.** Soft training may use \(\lambda\)-penalties; hard feasibility is guaranteed whenever \(\Pi_{B_t}\) is applied. Theorem 1 is intentionally a **weak, exact** claim about projection—not about task return.

**Detailed write-up:** [`theory/theorem_budget.md`](theory/theorem_budget.md).

---

## Algorithm 1 — Constraint-Aware Dynamic Graph Decision

```
Require:
  state s_t
  budget B_t                          # static case: B_t ≡ B
  candidate support A_t
  topology policy φ_θ
  physical policy π_ψ
  message map M

Output:
  feasible graph G_t, action a_t

1  Estimate edge utility     S_t ← φ_θ(s_t)
2  Project to feasible set   G_t ← Π_{B_t}(S_t)     # constraint projection operator
3  Aggregate messages        m_t ← M(s_t, G_t)
4  Compute physical action   a_t ← π_ψ(s_t, m_t)    # residual-corrected actor OK
5  Execute a_t; observe r_t, s_{t+1}
```

**Notes.**  
- Step 2 remains an **operator abstraction**; do not expand into sorting / knapsack / MIP in the main text.  
- Naming shift from RA-L: not “heuristic selection,” but **constraint projection** with Theorem 1.

---

## 5 Theoretical Analysis

We establish a three-link chain from budgeted topology decisions to shared-state return control:
\[
\boxed{
\text{Budget}
\;\rightarrow\;
\text{Topology distance}
\;\rightarrow\;
\text{Information loss}
\;\rightarrow\;
\text{Return bound}
}
\]
equivalently \(\textbf{Constraint Feasibility}+\textbf{Topology-induced Performance Stability}\).  
Full proofs: [`theorem_budget.md`](theory/theorem_budget.md), [`lemma_information_discrepancy.md`](theory/lemma_information_discrepancy.md), [`theorem_return_bound.md`](theory/theorem_return_bound.md).

### 5.1 Dynamic Topology Projection Feasibility

**Definition (Feasible graph set).**  
\[
\mathcal{G}_{B_t}=\{G\in\mathcal{G}:C(G)\le B_t\}.
\]
Static budgets are the special case \(B_t\equiv B\).

**Theorem 1 (Feasibility of budget-constrained dynamic topology projection).**  
Let \(S_t=\phi_\theta(s_t)\) and \(G_t=\Pi_{B_t}(S_t)\). Then:
1. **Feasibility:** \(C(G_t)\le B_t\).  
2. **Utility-maximizing projection:** \(G_t\) maximizes \(\sum_{(i,j)\in E(G)}g_{ij}\) over \(\mathcal{G}_{B_t}\)—**not** \(\arg\max_G J(\pi_\psi,G)\).

### 5.2 Topology-Induced Information Discrepancy

**Full-information communication reference** (not an optimum): \(G_t^\star=G_{\mathrm{full}}\).  
Graph distance: \(d_G(G_t,G_t^\star)=\|A_t-A_t^\star\|_F\).

**Assumption M.** Message map Lipschitz in topology: \(\|M(G_1)-M(G_2)\|\le L_M\,d_G(G_1,G_2)\).

**Lemma 2 (Topology sparsification induces bounded information discrepancy).**  
For \(G_t=\Pi_{B_t}(S_t)\) from Theorem 1,
\[
\boxed{
\varepsilon_G(t)
=
\|M(G_t^\star)-M(G_t)\|
\;\le\;
L_M\,d_G(G_t,G_t^\star)
=
L_M\|A_t-A_t^\star\|_F.
}
\]
Hence finite \(N\) \(\Rightarrow\) finite \(\varepsilon_G\). Lemma 2 **bridges** feasible projection (Thm.~1) to the discrepancy primitive used by Theorem 2. It does **not** claim that \(\varepsilon_G\) is task-optimal or vanishing.

### 5.3 Shared-State Performance Stability

**Shared-state (twin) evaluation.** Fix common \(\{s_t\}\) and
\[
J_T^\star=\sum_{t=0}^{T}\gamma^t r(s_t,a_t^\star),
\qquad
J_T=\sum_{t=0}^{T}\gamma^t r(s_t,a_t),
\]
with \(a_t^\star=\pi_\psi(s_t,M(G_t^\star))\), \(a_t=\pi_\psi(s_t,M(G_t))\).  
This is discounted cumulative reward under **fixed trajectory coupling**—not \(\rho^\star\approx\rho\).

**Assumptions A2–A3.** Policy / reward Lipschitz with constants \(L_\pi\), \(L_R\).  
(A1 of Theorem 2 is instantiated by Lemma 2: \(\varepsilon_G\le L_M\sup_t d_G\).)

**Lemma 1 (Topology-induced action gap).**  
\[
\|a_t^\star-a_t\|\le L_\pi\varepsilon_G.
\]

**Theorem 2 (Discounted return bound under shared-state topology approximation).**  
\[
\boxed{
|J_T^\star-J_T|
\le
\frac{L_R L_\pi\varepsilon_G}{1-\gamma}
\le
\frac{L_R L_\pi L_M}{1-\gamma}\sup_t d_G(G_t,G_t^\star).
}
\]

**Interpretation.** Topology-induced information discrepancy yields bounded shared-state return discrepancy. The bound characterizes topology-induced degradation under shared-state coupling, while closed-loop guarantees would require additional transition regularity assumptions—which we do **not** introduce. Does **not** prove closed-loop distributional equivalence, global optimality, or learning convergence.

**Corollary 1 (Residual recovery).**  
If \(a=\pi_\psi(s,m)+\beta\Delta\) with residual gap \(\varepsilon_\Delta\),
\[
\boxed{
\|a^\star-a\|\le L_\pi\varepsilon_G+\beta\varepsilon_\Delta,
}
\qquad
\boxed{
|J^\star-J|
\le
\frac{L_R(L_\pi\varepsilon_G+\beta\varepsilon_\Delta)}{1-\gamma}.
}
\]

### 5.4 Proposition — Communication Complexity under Fixed-Degree Projection

**Proposition (Communication complexity under fixed-degree projection).**  
If \(G=\Pi_B(S)\) satisfies \(\max_i d_i(G)\le K\), then \(\lvert E(G)\rvert\le NK\), hence \(C(G)=O(NK)\). When \(K=O(1)\),
\[
C(G)=O(N).
\]
**Proof.** Sum per-agent degree caps. □  

This is a direct consequence of the degree-budget specialization of Theorem 1. It does **not** assert performance preservation or generalization across swarm sizes.  
Detailed write-up: [`theory/proposition_complexity.md`](theory/proposition_complexity.md).  
*(Strong Theorem 3 cancelled — see [`theory/theorem_scalability.md`](theory/theorem_scalability.md).)*

### 5.5 Theory status

| ID | Role | Status |
|----|------|--------|
| Thm.~1 | Budget feasibility + utility-maximizing projection | **frozen** |
| Lem.~2 | Projection \(\rightarrow\) bounded \(\varepsilon_G\) | **merged** |
| Lem.~1 / Thm.~2 / Cor.~1 | Shared-state return stability | **frozen** |
| Prop. | Communication complexity under fixed-degree projection | **frozen** |
| Thm.~3 | Graph-size performance / generalization | **cancelled** |

## 6 Experiments

**Status:** design frozen — see [`experiments/tro_experimental_design.md`](experiments/tro_experimental_design.md).  
**Implementation protocols:** [`tro_logging_protocol.md`](experiments/tro_logging_protocol.md), [`tro_evaluation_protocol.md`](experiments/tro_evaluation_protocol.md), [`tro_reproducibility_checklist.md`](experiments/tro_reproducibility_checklist.md).  
**§6.3 status:** **FROZEN** with formal multi-seed evidence ([`evidence_6_3_formal/`](experiments/evidence_6_3_formal/)). Experiments report trends *consistent with* theory; they do not prove lemmas/theorems.

Organizing principle: theory-**aligned** empirical examination of
\[
\phi_\theta(s)\;\rightarrow\;G_t=\Pi_{B_t}(S_t)\;\rightarrow\;C(G_t)\le B_t
\;\rightarrow\;D_G\;\rightarrow\;\varepsilon_G\;\rightarrow\;\Delta A_\gamma,
\]
not a generic “compare against baselines” bake-off.

### 6.1 Experimental Setup

- **Tasks:** Cooperative Navigation (T1; used in §6.3); UAV Coverage / Exploration (T2); Target Tracking (T3).  
- **Backbone:** frozen AC-DSGF + MAPPO; hard projection \(\Pi_{B_t}\) at evaluation.  
- **Baselines (three classes; §6.6):** (A) IPPO / MAPPO; (B) CommNet, GAT-MAPPO, DGN; (C) TarMAC, IC3Net, ATOC (, MAGIC).  
- **Metrics:** \(\rho\), \(\bar d\), \(V_B\), \(D_G\), \(\varepsilon_G\), \(\Delta a\), \(\Delta A_\gamma=\sum_t\gamma^t\|a_t^\star-a_t\|\), \(C\), success, return \(J\).

### 6.2 Budget-Constrained Topology Projection Analysis
**Status: FROZEN (formal table).**

*Supports Theorem 1 (feasibility — not task performance).*  
Protocol: [`experiments/tro_6_2_formal_protocol.md`](experiments/tro_6_2_formal_protocol.md).  
We examine whether the hard projection \(G_t=\Pi_{B_t}(S_t)\) consistently satisfies \(C(G_t)\le B_t\) across swarm sizes and budget parameterizations. This section is **not** a performance comparison (cf. §6.3 / §6.6).

**Factors.** \(N\in\{8,16,32,64\}\); fixed degree \(K\in\{1,2,4,6,8\}\); ratio budgets \(\rho_B\in\{0.05,0.1,0.2,0.4\}\); five frozen `uav16` checkpoints; no training. For \(N\le 16\) we use closed-loop VMAS rollouts; for \(N\in\{32,64\}\) the same frozen scorer + hard \(\Pi_{B_t}\) is evaluated on geometric position samples (projection operator unchanged).

**Metrics.** Violation rate \(VR=\frac{1}{T}\sum_t\mathbf{1}(V_B(t)>0)\), maximum violation \(V_{\max}=\max_t V_B(t)\), mean degree \(\bar d\), density \(\rho_t=\lvert E_t\rvert/[N(N-1)]\). Under fixed-\(K\), the feasible cap is \(k_i=\min(K,\lvert\mathcal{N}_i\rvert)\); when the radius neighborhood is smaller than \(K\), \(\bar d\) tracks \(\overline{k_{\mathrm{cap}}}\) rather than unconstrained \(K\).

**Result.** Across all \((N,K)\) and \((N,\rho_B)\) and all five seeds, \(V_{\max}=0\) and \(VR=0\). Full grids: [`experiments/evidence_6_2_budget/`](experiments/evidence_6_2_budget/). Fig.~4 ([`figures/Fig4_communication_scaling.png`](figures/Fig4_communication_scaling.png)) shows \(\rho\) decreasing with \(N\) under fixed \(K\), without a scalability claim.

**Table I.** Budget feasibility under different swarm sizes and constraints (seed-aggregated; representative fixed-\(K\) slices). Every ratio-budget cell likewise reports \(V_{\max}=VR=0\).

| \(N\) | \(K\) | Budget type | Avg degree | Max violation | Violation rate |
|------:|------:|-------------|-----------:|--------------:|---------------:|
| 8 | 2 | fixed-\(K\) | 1.023 | 0 | 0 |
| 16 | 4 | fixed-\(K\) | 1.209 | 0 | 0 |
| 32 | 6 | fixed-\(K\) | 1.352 | 0 | 0 |
| 64 | 8 | fixed-\(K\) | 1.285 | 0 | 0 |

> The projection layer consistently satisfies the predefined communication budgets across different swarm sizes and constraints, demonstrating the practical feasibility of the proposed constrained topology decision mechanism.

### 6.3 Topology-Induced Information and Performance Analysis
**Status: FROZEN.**

We empirically examine whether the learned topology exhibits the theoretically characterized relationships among topology deviation, information discrepancy, and action deviation. Throughout, \(G_t^\star\) denotes the **full-support reference topology before budget projection** (pre-projection scores on the geometric support), and \(G_t=\Pi_{B_t}(S_t)\) is the projected sparse topology. Evaluation uses five frozen `uav16` checkpoints (seeds \(1234,2026,3407,42,8888\)), \(K\in\{1,2,4,6\}\) (chosen to avoid radius-neighborhood saturation), and \(32\) episodes per seed. Figures: [`figures/Fig1_topology_information.png`](figures/Fig1_topology_information.png), [`Fig2_information_action.png`](figures/Fig2_information_action.png), [`Fig3_budget_performance.png`](figures/Fig3_budget_performance.png).

**Topology–information relation (Fig.~1).**  
Stronger sparsification (smaller \(K\)) increases topology deviation \(D_G=\|A_t-A_t^\star\|_F\). Message discrepancy \(\varepsilon_G=\|M(G_t^\star)-M(G_t)\|\) tracks \(D_G\) with an approximately linear empirical trend \(\varepsilon_G\approx\alpha D_G+\beta\) (\(\alpha\approx 0.72\), \(R^2\approx 0.94\) on \(K\)-means). The empirical relationship is **consistent with** the topology-induced information discrepancy characterized in Lemma~2; we do not claim a statistical identification of \(L_M\).

**Information–action relation (Fig.~2).**  
Under shared-state twin evaluation (E1), we measure instantaneous action gap \(\Delta a=\|a^\star-a\|\) and discounted action discrepancy
\[
\Delta A_\gamma
=
\sum_{t}\gamma^t\|a_t^\star-a_t\|.
\]
Both decrease monotonically as \(K\) increases and \(\varepsilon_G\) shrinks, **consistent with** Lemma~1.  
Since the twin evaluation isolates topology-induced information changes under identical states, we use discounted action discrepancy as the primary metric rather than task return discrepancy, which additionally depends on closed-loop state distribution shifts. This choice aligns Fig.~2 with the proof intermediate of Theorem~2 without over-claiming closed-loop return preservation.

**Budget–performance tradeoff (Fig.~3).**  
Closed-loop (E0) reward, success, and communication cost versus budget level (fixed-\(K\) and ratio schedules; mean\(\pm\)std over five seeds) exhibit an adaptive communication–performance Pareto: aggressive sparsification substantially reduces \(C\) while task reward remains in a comparable range to the full-support reference under the present navigation setting. Fig.~3 is a tradeoff illustration for the theory narrative; statistical method comparisons are deferred to §6.6.

### 6.4 Scalability Analysis with Increasing Swarm Size
**Status: FROZEN (empirical analysis; no Theorem 3).**

Protocol: [`experiments/tro_6_4_formal_protocol.md`](experiments/tro_6_4_formal_protocol.md).  
Evidence: [`experiments/evidence_6_4_scalability/`](experiments/evidence_6_4_scalability/).  
We study how communication cost and efficiency evolve with swarm size under a **fixed** degree budget \(K=4\) (no retraining; no retuning of \(K\) with \(N\)). This section is an **empirical scalability analysis**. It supports the complexity proposition in §5.4; it does **not** establish a performance-generalization theorem.

**Factors.** \(N\in\{16,32,64,128\}\); methods AC-DSGF (\(K=4\)), DSGF (sparse family), Full Attention (dense reference); five frozen seeds; 16 episodes per cell. Spawn geometry scales with \(N\); dynamics / reward / \(K\) stay fixed.

**Claim hygiene.** Episode return \(J_N\) grows with \(N\) because the aggregate task scale changes; success rates fall with \(N\) (task difficulty shifts). We therefore **do not** claim that “performance scales with swarm size” or that topology learning transfers from \(N=16\) to \(N=128\). Primary claims concern communication growth and efficiency.

**Observation 1 — Linear communication growth.**  
For AC-DSGF, \(C_N/N\) remains in a narrow band \(\approx 1.54\)–\(1.93\) as \(N:16\to128\), consistent with \(C_N=O(N)\) under fixed \(K\) (and with the complexity proposition). DSGF exhibits a similar band (\(\approx 1.47\)–\(2.10\)).

**Observation 2 — Sparse communication advantage.**  
Full Attention pays \(C=N(N-1)\) (\(C/N:15\to127\)), i.e.\ near-quadratic growth. At \(N=128\), AC-DSGF mean cost is \(\approx 247\) versus \(16256\) for Full Attention (Fig.~7).

**Observation 3 — Communication efficiency.**  
Efficiency \(\eta_N=J_N/C_N\) favors sparse methods: AC-DSGF \(\eta\) rises from \(\approx 0.65\) to \(\approx 1.16\), while Full Attention falls from \(\approx 0.07\) to \(\approx 0.016\) (Fig.~8). This evidences communication efficiency—not a scalability theorem for task return.

**Table III.** Seed-aggregated scalability statistics (mean over 5 seeds; fixed \(K=4\) for AC-DSGF).

| Method | \(N\) | \(J\) | Success | \(C\) | \(C/N\) | \(\eta=J/C\) |
|--------|------:|------:|--------:|------:|--------:|-------------:|
| AC-DSGF (\(K=4\)) | 16 | \(15.90\pm0.97\) | 0.212 | \(24.6\pm1.9\) | 1.54 | 0.651 |
| AC-DSGF (\(K=4\)) | 32 | \(41.08\pm3.93\) | 0.105 | \(60.4\pm6.8\) | 1.89 | 0.685 |
| AC-DSGF (\(K=4\)) | 64 | \(117.07\pm9.30\) | 0.052 | \(111.5\pm5.7\) | 1.74 | 1.052 |
| AC-DSGF (\(K=4\)) | 128 | \(287.00\pm25.57\) | 0.026 | \(247.2\pm14.6\) | 1.93 | 1.163 |
| DSGF | 16 | \(16.16\pm1.01\) | 0.202 | \(23.6\pm2.4\) | 1.47 | 0.690 |
| DSGF | 32 | \(41.09\pm2.96\) | 0.108 | \(60.2\pm7.4\) | 1.88 | 0.689 |
| DSGF | 64 | \(118.30\pm5.23\) | 0.049 | \(118.4\pm15.4\) | 1.85 | 1.012 |
| DSGF | 128 | \(289.08\pm17.40\) | 0.025 | \(269.3\pm32.5\) | 2.10 | 1.085 |
| Full Attention | 16 | \(16.82\pm1.86\) | 0.243 | \(240\) | 15.0 | 0.070 |
| Full Attention | 32 | \(41.45\pm6.21\) | 0.122 | \(992\) | 31.0 | 0.042 |
| Full Attention | 64 | \(115.47\pm15.79\) | 0.045 | \(4032\) | 63.0 | 0.029 |
| Full Attention | 128 | \(257.92\pm35.60\) | 0.016 | \(16256\) | 127.0 | 0.016 |

Figures: [`figures/Fig7_communication_scaling.png`](figures/Fig7_communication_scaling.png) (\(C_N\) vs \(N\)); [`figures/Fig8_efficiency_scaling.png`](figures/Fig8_efficiency_scaling.png) (\(\eta_N\) vs \(N\)).

> Under a fixed degree budget, the learned sparse topology exhibits approximately linear communication growth and markedly higher communication efficiency than dense attention (empirical)—without claiming closed-loop performance generalization across swarm sizes.

### 6.5 Performance under Imperfect Communication Channels
**Status: FROZEN.**

Protocol: [`experiments/tro_6_5_formal_protocol.md`](experiments/tro_6_5_formal_protocol.md).  
We stress-test frozen policies under imperfect channels **without** retraining (5 seeds × 16 episodes). Fig.~6: [`figures/Fig6_channel_robustness.png`](figures/Fig6_channel_robustness.png) (*filename historical; claim = performance stability, not a robustness theorem*).

**Methods.** AC-DSGF; DSGF (non-budgeted sparse family); Full Attention (dense).

**Packet loss.** Under \(p_\ell\in\{0,0.1,0.3,0.5\}\), AC-DSGF return remains in a narrow band (\(J\approx 8.9\)–\(9.3\)) while delivered cost falls (\(C: 43\rightarrow 22\)); success stays near \(0.23\). DSGF shows a similar \(J\) plateau. Full Attention retains higher absolute return but at \(C\in[120,240]\).

**Delay.** With \(\tau\in\{0,20,50,100\}\) ms (steps \(\{0,1,2,4\}\)), all three methods exhibit only mild \(J\) variation (no sharp collapse), consistent with short-horizon navigation dynamics under stale messages.

**Bandwidth.** Hard projection at \(B/B_{\mathrm{full}}\in\{0.1,0.2,0.4,0.8\}\) keeps AC-DSGF return near \(J\approx 9.1\) while \(C\) scales with the budget (\(15\rightarrow 42\)), aligning the channel stress test with Theorem~1's feasible set \(\mathcal{G}_{B_t}\).

> AC-DSGF maintains more stable coordination performance under degraded communication conditions.

Outputs: [`experiments/evidence_6_5_channel/`](experiments/evidence_6_5_channel/).

### 6.6 Comparison with Communication Learning Baselines
**Status:** Phase A **FROZEN** (corrected \(C\); Class C deferred).

Protocol: [`experiments/tro_6_6_formal_protocol.md`](experiments/tro_6_6_formal_protocol.md).  
We compare against **representative communication paradigms that correspond to different communication-cost regimes** under matched \(N=16\) navigation and identical seeds (5×32 episodes)—on the joint \((J,C)\) plane, not Success alone. Fig.~5: [`figures/Fig5_baseline_pareto.png`](figures/Fig5_baseline_pareto.png).

**Phase A methods (cost regimes).**  
(A) MAPPO — no learned communication (\(C=0\));  
(B) GAT-MAPPO, DSGF — radius-support aggregation without budget projection as a decision;  
(G2) Full Attention — dense ceiling (\(C=N(N-1)=240\));  
(Ours) AC-DSGF — topology policy with hard \(\Pi_{B_t}\), including fixed-\(K\in\{2,4,6\}\) slices.

**Table II.** Mean±std over five seeds (unified reeval).

| Class | Method | \(J\) | Success | \(C\) |
|-------|--------|------:|--------:|------:|
| A | MAPPO | \(9.62\pm0.64\) | \(0.272\pm0.03\) | \(0\) |
| B | GAT-MAPPO | \(8.78\pm0.52\) | \(0.221\pm0.03\) | \(44.4\pm2.4\) |
| B | DSGF | \(9.01\pm0.42\) | \(0.232\pm0.03\) | \(43.7\pm3.6\) |
| G2 | Full Attention | \(9.60\pm0.90\) | \(0.288\pm0.04\) | \(240\) |
| Ours | AC-DSGF (default) | \(8.96\pm0.66\) | \(0.227\pm0.03\) | \(43.5\pm1.3\) |
| Ours | AC-DSGF (\(K=2\)) | \(9.24\pm0.68\) | \(0.239\pm0.03\) | \(27.5\pm0.5\) |
| Ours | AC-DSGF (\(K=4\)) | \(9.14\pm0.73\) | \(0.254\pm0.03\) | \(40.9\pm1.1\) |
| Ours | AC-DSGF (\(K=6\)) | \(9.26\pm0.58\) | \(0.234\pm0.03\) | \(43.6\pm2.0\) |

**Reading.** On the \((C,J)\) plane, AC-DSGF at \(K=2\) matches Class-B return while cutting communication below the radius-support baselines; Full Attention raises cost by an order of magnitude without a commensurate return gain over MAPPO. The comparison axis is **topology decision under budget**, not attention-mechanism competition; we do not claim superiority over all message-learning methods.

## 7 Discussion (preview)

Raising topology to a decision variable reframes “communication-efficient MARL” as **constrained dynamic graph decision**. The center formula remains
\[
\boxed{G_t=\phi_\theta(s_t)},
\]
obtained online via utility estimation and hard projection \(\Pi_{B_t}\)—**not** post-training / post-hoc pruning of a fixed graph. Empirically, fixed-degree projection yields approximately linear communication growth (§6.4), matching the complexity proposition—without claiming swarm-size performance generalization. RA-L soft-activation evidence is complementary engineering context; it is not restated here as a T-RO theorem.

---

## References

Primary citations (seed + constrained / coordination anchors). Full notes: [`references/`](references/).

**Communication MARL.**  
1. Foerster et al., Learning to communicate with deep multi-agent RL, NeurIPS 2016.  
2. Sukhbaatar et al., CommNet, NeurIPS 2016.  
3. Das et al., TarMAC, ICML 2019.  
4. Jiang & Lu, ATOC, NeurIPS 2018.  
5. Singh et al., IC3Net, ICLR 2019.  

**Graph MARL / GNN.**  
6. Veličković et al., GAT, ICLR 2018.  
7. Jiang et al., DGN, ICLR 2020.  
8. Kipf & Welling, Semi-Supervised Classification with GCNs, ICLR 2017.  

**Constrained / resource-aware RL.**  
9. Altman, *Constrained Markov Decision Processes*, 1999.  
10. Achiam et al., Constrained Policy Optimization, ICML 2017.  
11. Tessler et al., Reward Constrained Policy Optimization, ICLR 2019.  

**Coordination / platforms.**  
12. Yu et al., MAPPO, NeurIPS 2022.  
13. Lowe et al., MADDPG, NeurIPS 2017.  
14. Bettini et al., VMAS, arXiv:2207.03530.  

*(Camera-ready: use [`references/ac_dsgf_tro.bib`](references/ac_dsgf_tro.bib); notes in [`references/`](references/). Audit: [`audit/reference_audit.md`](audit/reference_audit.md).)*
