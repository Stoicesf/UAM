# 表 2 · 消融与机制验证

## 2a 静默 / 通信机制（\(N{=}16\)，诊断设定）

| Variant | Success (%) | Comm | CEI |
|---------|------------:|-----:|----:|
| AC-full（所学门控） | 24.6 | 0.008 | 32.0 |
| AC-no budget（\(g{=}A\)） | 28.9 | 40.6 | 0.007 |
| AC-random gates | 28.5 | 20.3 | 0.014 |

> 打开半径全边使 Comm 抬高三个数量级以上，Success 仅小幅上升 → 稀疏 ≠ 崩溃。

## 2b 残差消融（\(N{=}4\)，机制测试，**非表 1 规模**）

| Variant | Success (%) |
|---------|------------:|
| w/o Residual | 0.22 |
| Full DSGF | 9.27 |

来源：`table2_ablation.csv` / 英文 silence 表。
