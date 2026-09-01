# 表 4 · 可扩展性 / 预算趋势（诊断）

## 4a 预算扫描 Success（%）· \(N{=}16\) · seed 42 · 64-ep

| Budget \(\rho\) | GAT | DSGF | AC-DSGF |
|----------------:|----:|-----:|--------:|
| 100% | 21.3 | 21.7 | 27.1 |
| 50% | 18.2 | 21.6 | 25.2 |
| 10% | 16.9 | 21.7 | 24.2 |

> 绝对 % **不可与表 1 混读**；报告 graceful degradation 趋势。

## 4b 复杂度随 \(N\)

| \(N\) | GAT / DSGF | AC-DSGF |
|------:|------------|---------|
| 4, 8 | \(\mathcal{O}(Nk)\) | \(\mathcal{O}(NK)\)，\(K\ll k\) |
| 16 | 实测见 Table3 | 实测见 Table3 |

来源：`table_budget_sweep16.csv` + complexity.tex。
