# DSGF-HRL: Method Definition (Draft for IEEE Submission)

> Status: Week 1 — mathematical design frozen; implementation follows in Week 2–3.
> GAT-MAPPO is frozen as baseline (20k early-stop config); no further λ tuning.

---

## I. Problem Definition

### Dynamic Communication-aware Multi-Agent Coordination (DC-MAC)

Consider $N$ UAVs in an unknown 2D workspace. At time $t$, agent $i$ observes:

$$
o_i(t) \in \mathcal{O}, \quad o_i = [p_i, v_i, g_i, \ell_i]
$$

where $p_i \in \mathbb{R}^2$ is position, $v_i \in \mathbb{R}^2$ velocity,
$g_i$ goal-relative vector, $\ell_i$ local range readings (lidar).

**Constraints:**

1. **Partial observability** — no global state $s(t)$ is available to any single agent.
2. **Dynamic communication topology** — graph $G_t = (\mathcal{V}, \mathcal{E}_t)$ changes as agents move.
3. **Long-range coordination** — task success requires information beyond 1-hop neighbors (bottlenecks, goal allocation).

**Objective:** maximize expected team return while minimizing collisions and communication cost:

$$
J = \mathbb{E}\left[\sum_t \sum_i \left( r_i^{\text{goal}}(t) + r_i^{\text{coll}}(t) \right) - \lambda_c \cdot C(G_t) \right]
$$

---

## II. Empirical Motivation (from Stage 1–2 experiments)

| Method | Frames | Success | Alignment | Train Reward |
|--------|--------|---------|-----------|--------------|
| MAPPO+MLP Guide | 10k | 2.5% | 0.81 | −0.03 |
| MAPPO+MLP Guide | 102k | 0.3% | 0.58 | 1.20 |
| MAPPO+GAT Guide | 20k | **4.7%** | 0.70 | 1.17 |
| MAPPO+GAT Guide | 102k | 0.2% | 0.66 | 2.85 |

**Finding:** Reward optimization ≠ task generalization. Graph-based guidance suffers from **over-specialization during prolonged optimization** — alignment and training reward increase while eval success degrades.

**Paper claim (Motivation):**

> Existing graph-based coordination policies optimize surrogate alignment objectives that diverge from task-level generalization under extended training.

DSGF addresses: (i) long-horizon policy drift, (ii) static neighbor aggregation in GAT, (iii) missing temporal coordination memory.

---

## III. Proposed Framework: DSGF-HRL

### Hierarchical decomposition

$$
\underbrace{\Phi_i(t) = \text{DSGF}(o_i, G_t, \mathcal{H}_t)}_{\text{global coordination field}}
\quad \rightarrow \quad
\underbrace{a_i(t) \sim \pi_\theta(a_i \mid o_i, \Phi_i)}_{\text{local NavRL policy (MAPPO)}}
$$

Critic remains centralized on raw observations (CTDE): $V_\phi(\mathbf{o})$.

Total reward:

$$
r_i = r_i^{\text{goal}} + r_i^{\text{coll}} + \lambda_g(t) \cdot \cos(\angle a_i, \angle \Phi_i)
$$

with adaptive $\lambda_g(t)$ (warmup + decay + alignment scaling) — inherited from Stage 1 ablation, applied only at DSGF training stage.

---

## IV. Module 1: Dynamic Graph Generator (Innovation 1)

### Naive baseline (current GAT)

$$
A_{ij}(t) = \mathbb{1}\left[\|p_i - p_j\| < R_c\right]
$$

### DSGF: quality-aware dynamic adjacency

Define pairwise communication quality:

$$
q_{ij}(t) = \exp\left(-\frac{\|p_i - p_j\|^2}{2\sigma^2}\right) \cdot \eta_{ij}(t)
$$

where $\eta_{ij}$ captures relative velocity alignment (approaching vs receding):

$$
\eta_{ij} = \sigma\left(\frac{v_i \cdot (p_j - p_i)}{\|v_i\| \|p_j - p_i\| + \epsilon}\right)
$$

**Weighted adjacency:**

$$
\tilde{A}_{ij}(t) = A_{ij}(t) \cdot q_{ij}(t), \quad A_{ij}(t) = \mathbb{1}\left[\|p_i - p_j\| < R_c,\; i \neq j\right]
$$

**Communication cost (for Experiment 2):**

$$
C(G_t) = \sum_{i<j} A_{ij}(t)
$$

Complexity: $O(N^2)$ graph construction, but attention restricted to $k$ neighbors → $O(kN)$ message passing.

---

## V. Module 2: Sparse Spatial Attention (Innovation 2)

Given node embeddings $h_i^{(0)} = W_o \cdot x_i$ where $x_i = [p_i, v_i, g_i, c_i]$:

$$
e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^\top [W h_i \| W h_j]\right)
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij}) \cdot \tilde{A}_{ij}}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik}) \cdot \tilde{A}_{ik}}, \quad \mathcal{N}_i = \{j : A_{ij}=1\}
$$

$$
h_i' = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} W h_j\right)
$$

**Difference from GAT-MAPPO:**

| | GAT-MAPPO | DSGF |
|---|-----------|------|
| Graph weights | binary $A_{ij}$ | quality-weighted $\tilde{A}_{ij}$ |
| Attention | static per step | $\alpha_{ij}^t$ conditioned on $(G_t, v, q)$ |
| Temporal | none | memory $\mathcal{H}_t$ (Module 3) |
| Output use | direct policy bias | fused guidance field $\Phi_i$ |

Stack $L$ layers ($L=2$ default). Multi-head extension: $H=4$ heads, concat → project.

---

## VI. Module 3: Temporal Memory (Innovation 3)

Maintain per-agent history buffer $\mathcal{H}_i = \{h_i^{t-T+1}, \ldots, h_i^t\}$ with $T=5$.

**Temporal encoder (lightweight GRU or causal 1D conv):**

$$
z_i(t) = \text{GRU}(h_i'(t), z_i(t-1))
$$

or multi-head self-attention over time (causal mask):

$$
z_i(t) = \text{Attn}_\tau(h_i^{t-T+1:t})
$$

**Guidance field output:**

$$
\Phi_i(t) = [\hat{d}_x, \hat{d}_y, \rho_i, \pi_i] = \text{MLP}(z_i(t))
$$

where $(\hat{d}_x, \hat{d}_y)$ are normalized cooperative direction, $\rho_i$ local congestion risk, $\pi_i$ task priority.

---

## VII. Full Forward Pass (Network Structure)

```
Input:  x_i = [p_i, v_i, g_i, lidar_i]  ∀i ∈ {1..N}
              │
              ▼
┌─────────────────────────────────────┐
│  Dynamic Graph Generator            │
│  A_ij, q_ij, Ã_ij = f(p, v, R_c)   │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Node Embedding: h_i = φ(x_i)       │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Sparse Spatial Attention (×L)      │
│  h_i' = SSA(h, Ã)                   │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Temporal Memory (GRU / causal Attn)│
│  z_i = TE(h_i', H_i)                │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Guidance Head → Φ_i ∈ R^4          │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  MAPPO Actor: π(a_i | o_i, Φ_i)    │
│  MAPPO Critic: V(o)  [CTDE]        │
└─────────────────────────────────────┘
```

**Parameter budget target:** ≤ 1.5× GAT-MAPPO params (fair comparison).

---

## VIII. Baselines (frozen, no further tuning)

| ID | Method | Role |
|----|--------|------|
| B0 | IPPO | No coordination |
| B1 | MAPPO | Standard CTDE MARL |
| B2 | MAPPO + MLP Guide | Static local guidance (Stage 1) |
| B3 | MAPPO + GAT Guide | Graph baseline, 20k early-stop (`gat_a_lambda003`) |
| B4 | MAPPO + Full-Attention | Dense $O(N^2)$ ablation |
| **Ours** | **DSGF-HRL** | Proposed |

GAT 102k result retained as evidence of **training degradation**, not as deployable policy.

---

## IX. Experiments (IEEE minimum)

### Exp-1: Scalability
Train: N ∈ {4, 8, 16}. Test generalization: N ∈ {32, 64} (zero-shot graph size transfer if feasible).

Metrics: Success, Collision, Path Length, Wall Time.

### Exp-2: Communication Efficiency
Plot Success vs $C(G_t)$ under varying $R_c$. Compare MAPPO (implicit full obs via critic), GAT, DSGF.

### Exp-3: Dynamic Topology Robustness
$R_c \in \{10, 8, 5, 3\}$ (normalized to env scale). Success degradation curves.

### Exp-4: Ablation
- w/o Dynamic Graph ($\tilde{A} \equiv A$, no $q_{ij}$)
- w/o Sparse Attention (dense attention)
- w/o Temporal ($T=1$)
- w/o Guidance ($\Phi_i = 0$)

### Exp-5: Training Horizon Analysis
Frames ∈ {20k, 50k, 102k}: show DSGF maintains success while GAT degrades.

---

## X. Target Contributions (for Abstract/Intro)

**C1.** Dynamic sparse graph fusion module with communication-quality-aware topology for DC-MAC.

**C2.** Hierarchical guidance-enhanced MAPPO that injects spatiotemporal coordination knowledge into decentralized execution.

**C3.** Scalable UAV swarm navigation benchmark with systematic evaluation up to 64 agents and communication-efficiency analysis.

---

## XI. Implementation Mapping (Week 2–3)

| Paper module | Code path (planned) |
|--------------|---------------------|
| Dynamic Graph Generator | `models/dynamic_graph.py` |
| Sparse Spatial Attention | `models/sparse_attention.py` (extend `guidance/sparse_attention.py`) |
| Temporal Memory | `models/temporal_encoder.py` |
| DSGF fusion | `models/dsgf.py` |
| Training | `configs/experiments/exp4_dsfg.yaml` |

Existing stubs: `guidance/dsfg_encoder.py` (spatial only, no temporal, binary graph) → to be replaced by full DSGF.

---

## XII. 8-Week Schedule (aligned)

| Week | Deliverable |
|------|-------------|
| 1 | This document + Method section draft ✅ |
| 2–3 | Dynamic graph + sparse attention, 4-UAV smoke |
| 4 | Full DSGF + temporal, 4-UAV training |
| 5 | Scalability 4/8/16 |
| 6 | Zero-shot 32/64 test |
| 7 | Ablations + training-horizon analysis |
| 8 | Paper first draft |

Demo/video deferred to Week 8+ (supplementary material only).
