# References — Communication MARL

**Role in paper:** Position prior work as *message / protocol / attention learning on largely fixed supports*.  
**Our distinction:** \(G_t=\phi_\theta(s_t)\) is an **explicit decision variable** under \(C(G_t)\le B_t\), not only *what* to send.

| Cite key | Reference | Object learned | Topology role |
|----------|-----------|----------------|---------------|
| Foerster2016 | Foerster et al., Learning to communicate with deep multi-agent RL, NeurIPS 2016 | communication protocol / messages | support largely fixed |
| Sukhbaatar2016 | Sukhbaatar et al., Learning Multiagent Communication with Backpropagation (CommNet), NeurIPS 2016 | continuous communication channels | dense / shared channel |
| Singh2019 | Singh et al., Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Domains (IC3Net), ICLR 2019 | gate *when* to communicate | binary gates; not hard \(B_t\) projection |
| Das2019 | Das et al., TarMAC: Targeted Multi-Agent Communication, ICML 2019 | attention *whom* to address | attention on candidate set; cost not hard feasible set |
| Jiang2018 | Jiang & Lu, Learning Attentional Communication for Multi-Agent Cooperation (ATOC), NeurIPS 2018 | attentional communication groups | scheduling / attention, not \(\Pi_{B_t}\) |
| Kim2019 | Kim et al., Learning to Schedule Communication in Multi-Agent Reinforcement Learning, ICLR 2019 | communication scheduling | schedule ≠ budgeted graph projection |
| Wang2020 | Wang et al., Learning Nearly Decomposable Value Functions via Communication Skewness, ICLR 2020 | when/what under sparsity pressure | soft sparsity, not hard \(\mathcal{G}_{B_t}\) |
| Freed2020 | Freed et al., Communication Learning via Differentiable Discrete Channels, AAMAS / related | discrete channel learning | channel coding focus |

## Positioning sentence (for Related Work)

Prior communication MARL primarily optimizes *message content*, *gating*, or *attention* on a predetermined or heuristically restricted support. Hard membership of \(G_t\) in a feasible family \(\mathcal{G}_{B_t}\) via a projection operator is typically secondary.

## Do / do not cite for

| Do | Do not |
|----|--------|
| Contrast “message learning vs topology decision” | Claim we outperform TarMAC/IC3Net/ATOC without matched runs |
| Explain Class C as same *paradigm family*, different axis | Call them “missing baselines that invalidate the paper” |
