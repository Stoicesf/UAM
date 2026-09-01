# T-RO Evaluation Protocol

**Paper:** AC_DSGF_TRO · Implementation Layer  
**Status:** **Frozen**  
**Depends on:** [`tro_experimental_design.md`](tro_experimental_design.md), [`tro_logging_protocol.md`](tro_logging_protocol.md)

---

## 0. Goal

Make every §6 claim traceable to a **named evaluation mode** and logged fields—so theory (shared-state) and practice (closed-loop) are never conflated.

---

## 1. Evaluation modes

| Mode | State | Topology | Use for |
|------|-------|----------|---------|
| **E0 Hard-proj rollout** | closed-loop \(s_{t+1}=P(s_t,a_t)\) | \(G_t=\Pi_{B_t}(S_t)\) | §6.2 feasibility, §6.4–6.6 task metrics |
| **E1 Twin shared-state** | fixed \(\{s_t\}\) from one rollout (or logged buffer) | both \(G^\star\) and \(G_t\) on same \(s_t\) | §6.3 Fig.~1–2, Lemma 1 / Thm.~2 |
| **E2 Closed-loop gap (secondary)** | independent rollouts | sparse vs full | optional appendix; **not** primary Thm.~2 evidence |

**Default for theory figures:** E1.  
**Default for task tables:** E0.

---

## 2. Twin pipeline (E1) — mandatory for §6.3

```
1. Collect or freeze state sequence {s_t}_{t=0..T}   # from E0 sparse OR full; document which
2. For each t:
     S_t      ← φ_θ(s_t)
     G_t      ← Π_{B_t}(S_t)
     G*_t     ← G_full
     M, M*    ← M(G_t;s_t), M(G*_t;s_t)
     ε_G(t)   ← ||M* - M||
     D_G(t)   ← ||A_t - A*_t||_F
     a, a*    ← π_ψ(s_t, M), π_ψ(s_t, M*)
     r, r*    ← R(s_t, a), R(s_t, a*)
3. J_T, J*_T  ← discounted sums of r, r*
4. Emit Fig.1–2 from logged (D_G, ε_G, ΔJ)
```

**Do not** run two independent agents and call the reward gap “Theorem 2 validation.”

---

## 3. Phase execution order

### Phase 1 — §6.2 Budget feasibility (gate)

**Mode:** E0  
**Pass:** \(V_B=0\) always; \(\bar d\approx K\); \(\rho\sim O(1/N)\) for fixed \(K\).  
**Fail action:** stop; fix projection / logging; do not tune reward.

**Outputs:** Fig A (\(V_B\)), Fig B (degree), Fig C (density).

### Phase 2 — §6.3 Theory core

**Mode:** E1 (primary), E0 for Fig.~3 task reward if needed  
**Outputs:** Fig 1–3.  
Optional: overlay illustrative bound \(L_R L_\pi\varepsilon_G/(1-\gamma)\) with explicit non-fit caption.

### Phase 3 — §6.4 Scalability (observe)

**Mode:** E0  
**Record:** \((N,C_N,J_N,SE_N)\).  
**No Thm.~3 text** (cancelled). §6.4 is empirical + complexity proposition.

### Phase 4 — §6.5 Channel robustness

**Mode:** E0 under \((p_\ell,\tau,B_t)\) grid.  
Log topology adaptation (`rho_t`, degrees) under impairment.

### Phase 5 — §6.6 Baselines

Two comparison groups (locked):

| Group | Methods | Purpose |
|-------|---------|---------|
| **G1 Identical budget** | AC-DSGF, TarMAC, IC3Net, MAGIC (and peers under matched \(C\) or \(K\)) | fair constrained comparison |
| **G2 Performance ceiling** | Full communication | cost–performance upper reference |

Never claim G1 wins by silently giving others a different budget.

---

## 4. Seeds, CI, aggregation

| Item | Spec |
|------|------|
| Seeds | ≥ 3; prefer **5** for main tables |
| Point estimate | mean across seeds |
| Uncertainty | ± std or 95% bootstrap CI (state which) |
| Episode count | fixed per seed (document in `meta.json`) |
| Fig.~1–2 points | per-episode means or budget-fraction aggregates; no cherry-picked timesteps |

---

## 5. Metric computation recipes

```text
rho_t     = E_count / (N * (N - 1))          # if directed
V_B       = max(0, C_t - B_t)
D_G       = frobenius(A_t - A_star)
epsilon_G = l2(M_full - M_sparse)
delta_a   = l2(a_star - a_sparse)
J_T       = sum_t gamma**t * r_sparse[t]
J_T_star  = sum_t gamma**t * r_star[t]
Delta_J   = abs(J_T_star - J_T)
SE_N      = J_N / max(C_N, eps)
```

Norms and directedness must match `meta.json`.

---

## 6. Forbidden evaluation practices

- Changing reward / scorer / budget definition after seeing results  
- Dropping failed tasks from tables without disclosure  
- Mixing soft-gate SCA with hard \(V_B\) without labeling  
- Using E2 closed-loop \(\Delta J\) as primary Thm.~2 support  
- Unequal message dimension / budget across G1 baselines without reporting

---

## 7. Caption templates (claim hygiene)

- Fig.~1: “Empirical association consistent with Lemma 2; not a proof.”  
- Fig.~2: “Twin shared-state return gap vs \(\varepsilon_G\); dashed curve is an illustrative Lipschitz envelope, not a fitted constant.”  
- Fig.~3: “Budget–performance tradeoff under E0/E1 as specified.”
