# References — Topology / Graph Structure Learning (adjacent)

**Role in paper:** Adjacent line that *learns or sparsifies graphs*, clarifying we are not claiming the first “learned graph,” but the first framing here as **budgeted coordination variable in a DGDP**.  
**Caution:** Do not over-claim novelty against every graph-learning paper; emphasize **hard projection + joint \(\Pi_{\theta,\psi}\)** + theory chain.

| Cite key | Reference | Focus | Contrast |
|----------|-----------|-------|----------|
| Franceschi2019 | Franceschi et al., Learning Discrete Structures for Graph Neural Networks, ICML 2019 | learn graph for GNN | usually supervised / not MARL budget \(\Pi_{B_t}\) |
| Kipf2018 | Kipf et al., Neural Relational Inference, ICML 2018 | infer interaction graphs | latent structure inference ≠ hard \(B_t\) control |
| SoftSparsification | post-hoc pruning / SCA-style sparsification (incl. RA-L track context) | prune existing edges | engineering complement; **not** restated as T-RO theorem |
| Graphon / large-graph | optional large-graph limits | asymptotic structure | we **do not** claim graphon generalization (Thm.~3 cancelled) |

## Positioning sentence

Learning or inferring interaction structure is an active topic. Our contribution is to cast \(G_t\) as a **resource-constrained coordination decision** with an explicit projection onto \(\mathcal{G}_{B_t}\), coupled to physical policies, and to analyze feasibility, shared-state discrepancy, and communication complexity—without claiming task-optimal topology or swarm-size performance generalization.

## RA-L note (locked)

RA-L SCA-sparsification evidence is **complementary engineering context**. Do not merge its narrative claims into this T-RO manuscript without re-deriving them in the DGDP framing.
