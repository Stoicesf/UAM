# T-RO Logging Protocol

**Paper:** AC_DSGF_TRO · Implementation Layer  
**Status:** **Frozen** — every theory symbol must be reconstructible from logs  
**Depends on:** [`tro_experimental_design.md`](tro_experimental_design.md)  
**Rule:** If a quantity appears in §5, it must appear in the log schema. No figure may use unreproducible ad-hoc recomputation.

---

## 0. Principles

1. **Observability.** \(G_t\), \(S_t\), \(B_t\), \(M(G_t)\), \(M(G^\star)\), \(\varepsilon_G\), \(a_t\), \(a_t^\star\), \(J_T\), \(J_T^\star\) are first-class logged fields.  
2. **Twin before closed-loop.** Shared-state coupling is the primary protocol for Lemma 2 / Theorem 2 figures.  
3. **Hard projection at eval.** Soft gates may be logged for diagnostics; budget checks use post-\(\Pi_{B_t}\) adjacency.  
4. **No silent units.** Choose directed vs undirected once; document in run metadata.

---

## 1. Run metadata (once per run)

```json
{
  "run_id": "string",
  "seed": 0,
  "task_id": "T1|T2|T3",
  "N": 16,
  "budget_mode": "degree_K|edge_B|fraction",
  "K": 4,
  "B_t_schedule": "constant|dynamic",
  "B_full": null,
  "budget_fraction": 0.4,
  "gamma": 0.99,
  "graph_directed": true,
  "projection": "hard_Pi_B",
  "method": "AC-DSGF|...",
  "git_commit": "hash",
  "config_hash": "hash"
}
```

---

## 2. Per-timestep topology log (Theorem 1)

**Every** eval timestep \(t\):

```json
{
  "t": 0,
  "s_hash": "optional hash of state for twin join",
  "A_t": [[0,1,...], ...],
  "S_t": [[...], ...],
  "B_t": 64,
  "C_t": 48,
  "E_count": 48,
  "degrees": [2, 2, ...],
  "V_B": 0.0,
  "rho_t": 0.2,
  "mean_degree": 2.0
}
```

| Field | Theory link |
|-------|-------------|
| `A_t` | \(G_t\) adjacency after \(\Pi_{B_t}\) |
| `S_t` | \(\phi_\theta(s_t)\) scores (pre-projection) |
| `B_t`, `C_t`, `V_B` | \(C(G_t)\le B_t\), violation |
| `rho_t`, `mean_degree` | \(\rho\), \(\bar d\) |

**Storage note.** For large \(N\), store `A_t` as sparse COO `(i,j)` lists + shape; dense OK for \(N\le 64\).

---

## 3. Per-timestep message / discrepancy log (Lemma 2)

On the **same** state \(s_t\), compute both graphs:

| Branch | Graph | Message |
|--------|-------|---------|
| Sparse | \(G_t=\Pi_{B_t}(S_t)\) | \(M_{\mathrm{sparse}}=M(G_t;s_t)\) |
| Full reference | \(G^\star=G_{\mathrm{full}}\) | \(M_{\mathrm{full}}=M(G^\star;s_t)\) |

```json
{
  "t": 0,
  "M_full": [..],
  "M_sparse": [..],
  "epsilon_G": 0.0,
  "D_G": 0.0,
  "A_star_norm": "optional ||A*||_F"
}
```

| Field | Definition |
|-------|------------|
| `epsilon_G` | \(\varepsilon_G(t)=\|M_{\mathrm{full}}-M_{\mathrm{sparse}}\|\) (fix L2 unless metadata overrides) |
| `D_G` | \(d_G=\|A_t-A^\star\|_F\) |

**Must not** obtain \(M_{\mathrm{full}}\) from a different rolled-out state.

---

## 4. Per-timestep twin action log (Theorem 2 chain)

Same \(s_t\), same \(\pi_\psi\):

\[
a_t^\star=\pi_\psi(s_t,M(G^\star)),
\qquad
a_t=\pi_\psi(s_t,M(G_t)).
\]

```json
{
  "t": 0,
  "a_star": [..],
  "a_sparse": [..],
  "delta_a": 0.0,
  "r_star": 0.0,
  "r_sparse": 0.0,
  "delta_r": 0.0
}
```

| Field | Theory link |
|-------|-------------|
| `delta_a` | \(\|a^\star-a\|\) (Lemma 1) |
| `delta_r` | \(\lvert r(s,a^\star)-r(s,a)\rvert\) on shared state |

Optional residual:
```json
{
  "delta_Delta": 0.0,
  "beta": 1.0
}
```

---

## 5. Episode aggregates

```json
{
  "episode_id": 0,
  "J_T_star": 0.0,
  "J_T": 0.0,
  "Delta_J": 0.0,
  "gamma": 0.99,
  "T": 100,
  "mean_epsilon_G": 0.0,
  "max_V_B": 0.0,
  "mean_rho": 0.0,
  "closed_loop_J": null,
  "closed_loop_note": "optional; OUTSIDE Thm2 formal scope if set"
}
```

\[
J_T^\star=\sum_{t=0}^{T}\gamma^t r(s_t,a_t^\star),\quad
J_T=\sum_{t=0}^{T}\gamma^t r(s_t,a_t),\quad
\Delta J=\lvert J_T^\star-J_T\rvert.
\]

`closed_loop_J` may be logged separately but **must not** replace twin \(\Delta J\) in Fig.~2 primary panel.

---

## 6. Figure → field map (locked)

| Figure | X | Y | Required fields |
|--------|---|---|-----------------|
| Fig A (budget violation) | \(t\) or episode | \(V_B\) | `V_B` |
| Fig B (degree scaling) | \(N\) | \(\bar d\) | `mean_degree`, metadata `K`,`N` |
| Fig C (density) | \(N\) | \(\rho\) | `rho_t` / `mean_rho` |
| Fig 1 (Lem.~2) | `D_G` | `epsilon_G` | both per-\(t\) or episode means |
| Fig 2 (Thm.~2) | `epsilon_G` | `Delta_J` | twin aggregates; optional bound curve |
| Fig 3 (tradeoff) | budget fraction / \(B\) | `J` or twin \(J_T\) | metadata + returns |

**Optional Fig.~2 guide curve (not a fit):**
\[
\frac{L_R L_\pi\,\varepsilon_G}{1-\gamma}
\]
with explicitly stated placeholder or separately estimated Lipschitz constants; caption must say *illustrative upper envelope, not fitted identification*.

---

## 7. File layout (recommended)

```text
runs/<run_id>/
  meta.json
  steps.parquet          # or .npz: topology + messages + twin actions
  episodes.jsonl
  figures/               # generated only from above
```

Prefer columnar `parquet` for \(A_t\) sparse + vector fields.

---

## 8. Minimal schema validation checklist

Before trusting a run:

- [ ] `max(V_B)==0` under hard projection (or fail Phase 1)  
- [ ] `epsilon_G` computed on identical `s_hash` / shared state  
- [ ] `a_star` and `a_sparse` use same `pi_psi` weights  
- [ ] `Delta_J` reconstructed from logged `r_star`,`r_sparse`,`gamma` matches stored value  
- [ ] Fig scripts read only logged columns (no hidden recompute of \(M\) with different dropout)
