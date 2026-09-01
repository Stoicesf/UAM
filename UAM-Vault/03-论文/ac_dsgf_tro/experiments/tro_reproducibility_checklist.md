# T-RO Reproducibility Checklist

**Paper:** AC_DSGF_TRO  
**Use:** gate before writing any result sentence into EN/CN §6  
**Related:** [`tro_logging_protocol.md`](tro_logging_protocol.md), [`tro_evaluation_protocol.md`](tro_evaluation_protocol.md)

---

## A. Theory–log consistency

- [ ] Every §5 symbol has a log field (`G_t`/`A_t`, \(S_t\), \(B_t\), \(C_t\), \(V_B\), \(D_G\), \(\varepsilon_G\), \(a,a^\star\), \(J_T,J_T^\star\))  
- [ ] \(G^\star=G_{\mathrm{full}}\) labeled as **reference**, never “optimal graph” in code comments or captions  
- [ ] Hard \(\Pi_{B_t}\) used for \(V_B\) and primary topology stats  
- [ ] Twin E1 used for Fig.~1–2 primary panels  

## B. Run hygiene

- [ ] `meta.json`: seed, \(N\), \(K\)/budget fraction, \(\gamma\), git commit, config hash  
- [ ] ≥3 seeds (prefer 5) for main claims  
- [ ] CI / std method stated once and reused  
- [ ] No post-hoc exclusion of seeds without appendix note  

## C. Phase gates

- [ ] **Phase 1:** \(V_B\equiv 0\) before Phase 2 plots  
- [ ] **Phase 2:** Fig.~1–3 generated only from logged columns  
- [x] **Phase 3:** §6.4 frozen as empirical scalability; Thm.~3 cancelled; complexity proposition written  
- [ ] **Phase 5:** Group G1 identical-budget vs G2 full-comm separated in tables  

## D. Claim hygiene (paper text)

- [ ] “Consistent with Thm.~1 / Lem.~2 / Thm.~2” — not “proves”  
- [ ] No silent closed-loop as Thm.~2  
- [ ] No reward / budget / baseline set edits after peeking at results (or version-bump design doc)  
- [ ] Failure cases kept as ablations / failure analysis  

## E. Artifact freeze (per paper revision)

- [ ] Figure scripts pinned to schema version  
- [ ] Raw `runs/` retained for camera-ready  
- [ ] EN/CN §6 numbers match regenerated tables from the same `run_id`s  

---

## Frozen: do not change mid-campaign

| Frozen | Notes |
|--------|-------|
| Theory narrative (§3–5) | No edits to claim strength based on bad runs |
| Design doc hypotheses | Version bump required if changed |
| Logging schema field names | Additive OK; renames need migrator |
| Baseline set Classes A/B/C | Additions OK; silent removals forbidden |
| Budget definition \(C(G)\) | Must match Thm.~1 |

---

## Sign-off

| Role | Name / date | OK |
|------|-------------|-----|
| Logging schema | | ☐ |
| Twin E1 verified | | ☐ |
| Phase 1 gate | | ☐ |
| §6 draft numbers | | ☐ |
