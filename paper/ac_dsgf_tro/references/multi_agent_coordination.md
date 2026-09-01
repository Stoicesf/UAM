# References — Multi-Agent Coordination / Platforms

**Role in paper:** Problem setting — cooperative multi-agent / multi-UAV coordination under limited sensing and communication.  
**Our distinction:** Coordination structure (who talks to whom) is part of the decision, not only physical actions.

| Cite key | Reference | Focus | Link to us |
|----------|-----------|-------|------------|
| Lowe2017 | Lowe et al., Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG), NeurIPS 2017 | centralized training | MARL backdrop |
| Yu2022 | Yu et al., The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO), NeurIPS 2022 | strong cooperative baseline | Class A baseline |
| Rashid2018 | Rashid et al., QMIX, ICML 2018 | value factorization | cooperative MARL lineage |
| Samvelyan2019 | Samvelyan et al., The StarCraft Multi-Agent Challenge, AAMAS 2019 | benchmark culture | optional |
| Bettini2022 | Bettini et al., VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning, arXiv:2207.03530 | simulator | our experimental platform |
| Oroojlooy2023 | Oroojlooy & Hajinezhad, A Survey on Multi-Agent Reinforcement Learning, related surveys | survey | related-work breadth |
| UAVSwarmSurvey | multi-UAV / swarm robotics surveys (communication-limited formation, coverage) | domain motivation | Intro §1.1 |

## Positioning sentence

Classical multi-agent coordination and modern MARL learn *actions* (and sometimes values) under an assumed interaction graph. Under bandwidth, energy, and interference limits, the interaction graph itself becomes a constrained decision.
