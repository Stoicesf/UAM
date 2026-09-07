# Phase 1：安全-critical RISE（SC-RISE）调研

分支：`feature/theory-scrise`  
对照代码：[`models/transport/control/rise_controller.py`](../../models/transport/control/rise_controller.py)、[`models/transport/control/hierarchical.py`](../../models/transport/control/hierarchical.py)、env 风扰路径

## 1. 背景

RISE 用滤波误差 + 含 \(\mathrm{sign}(e)\) 的积分项抑制扰动，工程上已在有风场景通过 `_rise_payload_assist` 拉回载荷。安全-critical 目标是：**扰动抑制不得破坏 CBF/障碍约束**（位置、机间、工作空间），即鲁棒性与安全证书统一。

## 2. 相关工作要点（调研提纲）

- **RISE / 连续滑模类**：有限时间扰动估计、符号项抖振与低通近似。
- **CBF 与鲁棒控制融合**：QP 中把鲁棒补偿当作扰动界；或安全过滤器投影名义+RISE 输入。
- **难点**：\(\mathrm{sign}\) 使闭环非光滑，经典 CBF 条件需 Filippov / 光滑近似；积分状态增大系统维数。

## 3. 相对本仓库现状的差距

| 现有实现 | SC-RISE 缺口 |
|----------|------------|
| `RISEController.compute`：\(e=\dot e_x+(k_p/k_v)e_x\)，积分含 \(\mathrm{sign}(e)\) | 无安全约束；输出任意叠加到推力/协助速度 |
| hierarchical：有风时写 `env._rise_payload_assist` | 旁路式载荷速度注入，**不经** CBF-QP |
| 有风时 env 关闭软巡航，放大 RISE A/B 差异 | 利于验收，但非统一理论闭环 |
| `HybridSafetyShield` 作用在 `des_vel` | 与 RISE 串级，顺序固定，无联合证明 |
| phase2 验收：RISE 下 \(d<0.5\)，无 RISE \(d>1\) | 证明的是扰动抑制，**未证**安全边界不越界 |

## 4. Phase 2 理论问题清单

1. 将 RISE 状态增广进闭环；写清非光滑向量场假设。
2. 提出 SC-RISE：名义 UBF/PD + RISE 补偿，经安全过滤器（投影或 QP）保证 \(h\ge 0\)。
3. 推导 ISS（对扰动）且安全集正向不变；符号项用饱和/双曲正切作可证明光滑近似。
4. 明确与方向一 CSCBF、方向二 M-UBF 的接口：SC-RISE 只拥有 `rise_controller` + 过滤器胶水层。

## 5. 验收门槛

| 阶段 | 门槛 |
|------|------|
| Phase 1 | 融合架构草图 + 与现 RISE 旁路差距 |
| Phase 3–4 | 强风下 SC-RISE 守住 CBF；普通 RISE 可展示越界对照 |
| Phase 5 | 阶跃扰动下偏差小且约束始终满足 |

## 6. 论文章节指针

拟写入：“鲁棒安全控制”（RISE → SC-RISE → 风扰实验）。
