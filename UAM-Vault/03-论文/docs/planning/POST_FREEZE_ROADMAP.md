# AC-DSGF Post-Freeze Roadmap (6–12 months)
# Updated: 2026-07-17
# Status: LOCKED — no ++ algorithm search

```
        AC-DSGF v1（冻结）
        通信约束协同学习
               |
    ┌──────────┼──────────┐
    ↓          ↓          ↓
论文投稿    工程系统    下一代算法
    ↓          ↓          ↓
IEEE RA-L   ROS2/Gazebo  AC-DSGF 2.0
T-RO        实机验证     通信-行动联合规划
```

## Priority order (do not invert)

| Rank | Track | Now |
|------|-------|-----|
| **P0** | Paper v1 → RA-L / T-RO | **ACTIVE** |
| P1 | ROS2 + Gazebo demo (4 UAV) | after draft solid |
| P2 | AC-DSGF 2.0 (comm-aware motion) | after submit |
| — | AC-DSGF++ / emergent protocol | **archived** (S5 only) |

## Phase 1 (now ~ 1 month) — Paper landing
- No algorithm changes, no λ retune, no ++ revival
- Week 1: Abstract · Intro · Related · Math · Pseudocode
- Week 2: Experiment narrative · captions · Supplement S5
- Week 3–4: Cover letter · compile · submit package
- Optional: 32-UAV *eval-only* scalability table (complexity story; not Success chase)

### Theory to add (raise venue tier)
- Lemma-style: sparsification \(O(N^2)\to O(KN)\)
- Residual prevents topology / guidance collapse
- Soft performance-under-budget discussion (not full optimality proof)

## Phase 2 (1–3 months) — Engineering
- ROS2 ↔ PyTorch AC-DSGF ↔ Gazebo/PX4
- 4 UAV exploration demo: GAT vs AC-DSGF (packets, latency, trajectory)
- Strengthens T-RO path

## Phase 3 (3–6+ months) — Next algorithm
**AC-DSGF 2.0:** Communication-aware Motion Planning  
(not “when to send” utility — ++ already showed that risk)

```
Obs → World/State cue → Comm Planner ↔ Motion Planner → Action
```

Agents may *move closer* to obtain high-value information.

Emergent discrete protocols (VQ-VAE etc.): **later / high risk** — not now.

## Venue ladder
1. IEEE RA-L (primary near-term)
2. IEEE T-RO (+ theory + real/sim-robot)
3. NeurIPS/ICRA only with new learning paradigm (2.0+)

## Forbidden
- Continue AC-DSGF++ hyperparameter / U* search for main paper
- “One more module” before submission
- Claiming causal communication as verified

## Master freeze
`paper/docs/freeze/AC_DSGF_FINAL_FREEZE.md`  
Checklist: `paper/docs/freeze/PAPER_SUBMISSION_CHECKLIST.md`
