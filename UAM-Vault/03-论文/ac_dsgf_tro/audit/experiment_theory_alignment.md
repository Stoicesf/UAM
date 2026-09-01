# Experiment ↔ Theory Claim Alignment

## §6.2 Budget feasibility

| Required | Status |
|----------|--------|
| Claim = budget feasibility only | OK |
| Metrics \(V_B\), \(V_{\max}\), \(VR\) | OK (\(V_{\max}=VR=0\)) |
| No “better performance” | OK |
| Framing “consistent with / supports Thm.~1 feasibility” not “proves” | OK |

## §6.3 Topology–information–action

| Figure | Chain | Framing |
|-------|-------|---------|
| Fig.~1 | \(D_G\to\varepsilon_G\) | **consistent with** Lem.~2 |
| Fig.~2 | \(\varepsilon_G\to\Delta A_\gamma\) | **consistent with** Lem.~1; twin / shared-state |
| Fig.~3 | \(B\to(J,C)\) Pareto | tradeoff illustration; not Thm.~2 closed-loop proof |

Forbidden “validate theorem” — not used in frozen EN/CN body.

## §6.4 Scalability (empirical)

| Required | Status |
|----------|--------|
| Linear \(C_N\) / \(C_N/N\) | Obs.~1 |
| Sparse vs dense \(C\) | Obs.~2 |
| \(\eta=J/C\) | Obs.~3 |
| No performance generalization / no Thm.~3 | OK |
| Aligns with Prop. complexity | OK |

## §6.5 Channel stress

| Required | Status |
|----------|--------|
| No “robust guarantee” | OK after title soften |
| Allowed: stable performance under degradation | quote present |
| No new robustness theorem | OK |

## §6.6 Baselines — cost definition

| Method | \(C\) definition | Value @ \(N=16\) |
|--------|------------------|------------------|
| MAPPO | no learned edges | \(0\) |
| GAT / DSGF | radius-support edges | \(\sim44\) |
| Full Attention | \(N(N-1)\) | \(240\) |
| AC-DSGF | projected topology edges | \(\sim27\)–\(44\) by \(K\) |

No old-definition residue found in Table II. Class C deferred — no placeholder scores.
