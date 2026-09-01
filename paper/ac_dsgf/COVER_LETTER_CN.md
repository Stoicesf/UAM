# 中文 Cover Letter — AC-DSGF v1（定稿）

## 题目
面向通信约束多无人机蜂群协同的自适应拓扑学习

尊敬的编辑：

现有多无人机策略常将通信拓扑视为固定基础设施（全连接 / 半径 / \(k\) 近邻）；attention 权重改进聚合，但不等于带宽限制下的链路激活决策。本文将自适应通信拓扑学习表述为蜂群任务性能与通信代价的联合优化，并提出 **AC-DSGF**（Candidate Edge Scoring → Budget-Constrained Edge Selection → Residual Recovery），以软预算代理 \(\mathbb{E}[R-\lambda_c C]\) 学习自适应稀疏模式。我们给出度约束规模界 \(\eta_N=\mathcal{O}(1/N)\)，并将所学门控解释为预算边选择的近似/诱导解（非显式组合求解器）。在相同通信预算下验证：可比协同性能的同时显著降低通信激活（SCA）。

**贡献要点：**
1. 将自适应通信拓扑学习表述为蜂群任务性能与通信代价的联合优化问题。
2. 提出可微拓扑适应机制：预算正则 + 残差引导（Candidate Edge Scoring → Budget-Constrained Edge Selection → Residual Recovery）。
3. 验证在不同蜂群规模与预算化部署下，可比协同性能的同时显著降低通信激活（SCA）。

此致  
敬礼  
[作者]

**关键词：** 自适应通信拓扑；软预算约束；无人机蜂群协同；通信激活；多智能体强化学习
