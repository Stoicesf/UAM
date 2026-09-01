# T-RO LaTeX Migration Notes (pending)

**Do not edit scientific Markdown as a substitute for template work.**  
**Source of truth until compile:** `AC_DSGF_TRO_EN.md` + `supp/` + `references/ac_dsgf_tro.bib`.

## Recommended pipeline

```text
v0.11 markdown
        ↓
IEEE T-RO / Transactions LaTeX template (download from IEEE)
        ↓
compile
        ↓
visual inspection
        ↓
final PDF
```

No `IEEEtran` / T-RO class file is currently vendored in this repo — obtain the official template for camera-ready.

## Migration checklist

### 1. Figures
- Fig.~5 (\((J,C)\)), Fig.~7 (\(C_N\)), Fig.~8 (\(\eta\)): keep readable width; **short captions** (no theorem claims in captions).

### 2. Algorithm
- Preserve \(G_t=\phi_\theta(s_t)\) and \(\Pi_{B_t}\) — do **not** compress to \(G=\phi(s)\).

### 3. Equation / theorem numbering
- Align Theorem 1, Theorem 2, Proposition (complexity) with main-text references.
- Thm.~2 title must remain: *Discounted Return Bound under Shared-State Topology Approximation*.

### 4. Abstract
- Keep: *communication-efficient coordination under constrained communication budgets*.
- Do **not** restore: scalable / optimal / universal.

### 5. Supplementary
- Port `supp/AC_DSGF_TRO_supp_EN.md` (+ proofs) as PDF appendix or separate supp PDF per venue rules.

### 6. BibTeX
- `\bibliography{references/ac_dsgf_tro}` (or copy `.bib` into the LaTeX project).
