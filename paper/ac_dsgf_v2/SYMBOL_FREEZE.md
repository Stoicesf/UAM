# 符号冻结（Submission Freeze）

| 符号 | 唯一含义 | 禁止别称 |
|------|----------|----------|
| \(\delta_t\) | \(d_H(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1})\) 可行域失配 | “prediction error”（容量残差请写 \(\lvert\hat c-c\rvert\)） |
| \(\chi_t\) | \(d_H(\mathcal{B}_{t+1},\mathcal{B}_t)\) 约束漂移 | “environment uncertainty” |
| \(\epsilon_t\) | \(\|s_{t+1}-\hat s_{t+1}\|\) 状态预测误差 | 与 \(\delta\) 混用 |
| \(\mathrm{PI}_t\) | \(\delta_t/(\chi_t+\varepsilon)\) | — |
| \(\alpha_t\) | \(1/(1+\mathrm{PI}_t^2)\) 实现层鲁棒混合 | 第四条“贡献” |
| \(c^{\mathrm{mix}}\) | \(\alpha\hat c+(1-\alpha)c\) → **单次** \(\Pi\) | 两投影凸组合 |

主文已在 `sections/theory.tex` 加 Notation freeze 段。
