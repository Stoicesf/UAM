# References — Graph MARL / GNN

**Role in paper:** Support local aggregation, graph-structured policies, and the use of \(\phi_\theta\) / \(M(G)\) on relational structure.  
**Our distinction:** Most Graph MARL methods learn aggregators **on** a graph; we treat the graph itself as a **budgeted decision**.

| Cite key | Reference | Focus | Link to us |
|----------|-----------|-------|------------|
| Velickovic2018 | Veličković et al., Graph Attention Networks, ICLR 2018 | attention aggregation | aggregator class used in baselines |
| Jiang2020 | Jiang et al., Graph Convolutional Reinforcement Learning (DGN), ICLR 2020 | multi-agent GCN on local graphs | graph as input structure |
| Liu2020 | Liu et al., Multi-Agent Game Abstraction via Graph Attention Networks, related | attention over agents | still support-driven |
| Agarwal2020 | Agarwal et al., Learning Transferable Cooperative Behavior in Multi-Agent Teams, related | GNN policies | transfer / local interaction |
| Kipf2017 | Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, ICLR 2017 | GCN foundation | message-passing background |
| Battaglia2018 | Battaglia et al., Relational inductive biases, deep learning, and graph networks, arXiv:1806.01261 | relational inductive bias | justify graph-structured \(M\) |
| Bronstein2021 | Bronstein et al., Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges, arXiv | permutation / locality | optional theory-adjacent cite |

## Positioning sentence

Graph MARL and GNN policies provide strong inductive biases for local interaction, but typically assume a radius / \(k\)-NN / fixed adjacency. We keep the aggregator family and instead decide \(G_t\) under explicit budgets.
