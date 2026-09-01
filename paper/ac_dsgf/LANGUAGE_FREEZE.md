# AC-DSGF v1 — 定稿语言冻结清单（Consistency Freeze）

**状态：** 论文定稿一致性维护阶段（不改算法、不扩 ++、不堆实验）

---

## 锁定名称

| 项 | 定稿 |
|----|------|
| 英文题 | Learning Adaptive Topologies for Communication-Constrained UAV Swarm Coordination |
| 中文题 | 面向通信约束多无人机蜂群协同的自适应拓扑学习 |
| 方法 | **AC-DSGF** = Adaptive Constraint-aware Dynamic Sparse Graph Framework |
| 版本 | **v1** |

---

## 禁止写入主文

- AC-DSGF++ / causal utility / World Model / Intelligent Swarm Brain  
- SwarmOS / PX4 / Adapter / Runtime / Fault Injector（工程隔离）  
- “AC reduces communication overhead” **当** \(|E_{\mathrm{AC}}|=|E_{\mathrm{RULE}}|\)  
- 以 kbps / Soft Mass 作为**主科学贡献**  
- training convergence / 学习收敛定理（Prop.1 仅规模界）  
- outperform / superior Success  

---

## 强制主表述

> Under identical communication budgets, AC-DSGF achieves task-aware topology selection while maintaining task effectiveness.

Prop.1 句：

> Under a bounded neighborhood \(|E_t(i)|\le K\), \(\eta_N\le K/(N-1)=\mathcal{O}(1/N)\). The proposition characterizes communication density scaling rather than optimization convergence.

Prop.2 句：

> Residual recovery bounds action deviation \(\|a-a^\star\|\); it does **not** guarantee non-decreasing task reward.

---

## Attention vs Topology（一句话）

- Attention: *who is important* (feature weighting)  
- Topology: *who should communicate* (link decision)

---

## 主文路径

- 中文：`paper/ac_dsgf_cn/AC_DSGF_CN.md`  
- 英文：`paper/ac_dsgf/AC_DSGF_EN.md`  
- 贡献/亮点/Cover：本目录对应 `*_CN.md` / `.md`

**下一阶段：** Submission Readiness——详见 [`AC_DSGF_v1_submission_checklist.md`](AC_DSGF_v1_submission_checklist.md) 与 [`REVIEWER2_AUDIT.md`](REVIEWER2_AUDIT.md)。不再改变问题定义与贡献边界。


## 核心故事（终稿）

> Communication topology is elevated from fixed infrastructure to a soft-budget decision variable.

主优势：**SCA / soft sparsity**，不是 hard Top-K 最优邻居。

Prop.3 只允许 *interpreted as / approximate / induced*，禁止 *solves / optimality theorem*。

多 \(\lambda_c\) 重训 Pareto：可选 camera-ready；当前用冻结策略预算工作曲线（Fig.13）。


## 术语禁忌
- **禁止使用 SCAM**（英语俚语“骗局”联想）。统一：**SCA** = Soft Communication Activation，\(C_s=\sum g_{ij}\)。
