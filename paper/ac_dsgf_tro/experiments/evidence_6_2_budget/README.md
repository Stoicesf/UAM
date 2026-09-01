# §6.2 Budget Feasibility Evidence

**Status:** **FROZEN** · gate `VR=Vmax=0` on full grid · manuscript Table I filled (EN/CN v0.7)

**Protocol:** [`../tro_6_2_formal_protocol.md`](../tro_6_2_formal_protocol.md)

## Files

| File | Content |
|------|---------|
| `budget_statistics.csv` | per-seed rows (\(N\), fixed-\(K\) / ratio, \(\bar d\), \(\rho\), \(V_{\max}\), \(VR\)) |
| `degree_statistics.csv` | degree vs caps |
| `evidence_6_2_report.json` | aggregate Table I + `all_pass` |
| `figures/Fig4_communication_scaling.png` | \(\rho(N)\) (also under `paper/.../figures/`) |
| `collect_log.txt` | formal run log |

## Eval modes

| \(N\) | Mode |
|-------|------|
| 8, 16 | VMAS closed-loop (`env`) |
| 32, 64 | frozen scorer + geom samples (`geom`) |

## Pass criterion

\(V_{\max}=0\) and \(VR=0\) for every \((N,K)\) and \((N,\rho_B)\) — **passed**.

## Claim form

Practical feasibility of \(\Pi_{B_t}\) — **does not prove** Theorem 1.
