# §6.4 Scalability Observation Protocol

**Status:** **FROZEN** (2026-07-24)  
**Manuscript title:** Scalability Analysis with Increasing Swarm Size  
**Purpose:** Empirical analysis of \(C_N\), \(\rho_N\), \(\eta_N\) vs \(N\). **Not** theorem validation.  
**Thm.~3:** **cancelled** → use [`../theory/proposition_complexity.md`](../theory/proposition_complexity.md).

---

## Claim hygiene

Forbidden:
> scalable optimal coordination / validates generalization theorem / performance scales with swarm size

Allowed:
> Empirical scalability analysis under fixed degree budget \(K\).  
> Sparse learned topology provides approximately linear communication growth.  
> Communication efficiency \(\eta=J/C\) favors sparse methods.

---

## Questions (answered)

1. Does \(C_N\) grow as \(O(N)\) under fixed \(K\)? → **yes** (Obs.~1; \(C_N/N\approx1.5\)–\(1.9\) for AC-DSGF).  
2. Does \(J_N\) stay “stable”? → **not used as a claim** (\(J_N\) grows with task scale; success falls).  
3. Does \(\eta_N=J_N/C_N\) beat dense reference? → **yes** (Obs.~3).

---

## Factors (frozen)

| Item | Spec |
|------|------|
| \(N\) | \(\{16,32,64,128\}\) |
| Degree budget | **fixed \(K=4\)** |
| Seeds | \(\{1234,2026,3407,42,8888\}\) |
| Episodes / cell | **16** |
| Training | **none** |
| Methods | AC-DSGF (\(K=4\)); DSGF; Full Attention |

---

## Outputs (frozen)

```text
paper/ac_dsgf_tro/experiments/evidence_6_4_scalability/
├── README.md
├── scalability_statistics.csv
└── evidence_6_4_report.json
paper/ac_dsgf_tro/figures/Fig7_communication_scaling.png
paper/ac_dsgf_tro/figures/Fig8_efficiency_scaling.png
```

EN/CN §6.4 + Table III written and frozen.
