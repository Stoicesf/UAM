# References — Resource-Constrained / Constrained RL

**Role in paper:** Justify \(C(G_t)\le B_t\) as a **hard feasibility** requirement (projection / constrained optimization), not only Lagrangian soft penalties or post-hoc pruning.  
**Our distinction:** Soft training surrogates \(\lambda c(G)\) may be used, but evaluation enforces \(\Pi_{B_t}\).

| Cite key | Reference | Focus | Link to us |
|----------|-----------|-------|------------|
| Altman1999 | Altman, Constrained Markov Decision Processes, Chapman & Hall/CRC, 1999 | CMDP foundations | conceptual budget constraints |
| Achiam2017 | Achiam et al., Constrained Policy Optimization, ICML 2017 | safe / constrained policy optimization | soft vs hard constraint practice |
| Tessler2019 | Tessler et al., Reward Constrained Policy Optimization, ICLR 2019 | reward-constrained RL | Lagrangian surrogates |
| Chow2018 | Chow et al., A Lyapunov-based Approach to Safe Reinforcement Learning, NeurIPS 2018 | safety constraints | feasibility mindset |
| Paternain2019 | Paternain et al., Constrained Reinforcement Learning Has Zero Duality Gap, NeurIPS 2019 | duality / constrained RL | theory background (optional) |
| Liu2021 | Liu et al., Policy Learning with Constraints in Model-free Reinforcement Learning: A Survey, related surveys | constrained RL survey | related-work breadth |
| EventTrigComm | e.g. event-triggered / bandwidth-aware multi-agent communication papers | reduce rate on fixed graph | **post-hoc / trigger ≠** \(\Pi_{B_t}\) onto \(\mathcal{G}_{B_t}\) |

## Positioning sentence

Resource-aware MARL often reduces communication via penalties, gates, or event triggers on a given support. We instead define a feasible topology family \(\mathcal{G}_{B_t}\) and obtain \(G_t=\Pi_{B_t}(S_t)\), so hard budget feasibility is an operator property (Theorem 1), not only an optimization hope.
