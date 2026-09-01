# 表 3 · Runtime（CPU 推理，无再训练）

> \(N{=}16\)，50 timed steps。来源：`table5_compute_cost.csv`

| Method | Complexity（被注意） | Runtime (ms/step) | Comm（表 1） |
|--------|----------------------|------------------:|-------------:|
| GAT | \(\mathcal{O}(N^{2}d)\) / \(\Theta(Nk)\) | 1.66 | 38.8 |
| DSGF | \(\mathcal{O}(Nkd+Nd)\) | 3.35 | 39.3 |
| **AC-DSGF** | \(\mathcal{O}(NKd+Nd)\) | **3.94** | **0.43** |

门控仅增加约 0.6 ms（相对 DSGF），通信质量约降两个数量级。
