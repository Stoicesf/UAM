# Corollary 4 — Graceful Recovery

**Status:** FROZEN (SECDO v2)  
**Depends on:** Theorem 3

## Mechanism

Define
\[
\alpha_t=\frac{1}{1+\mathrm{PI}_t^2},\qquad
c_t^{\mathrm{mix}}=\alpha_t\hat c_{t+1}+(1-\alpha_t)c_t.
\]
When prediction fails (\(\mathrm{PI}_t\gg1\)), \(\alpha_t\to0\) and SECDO recovers reactive projection \(\Pi_{\mathcal{B}_t}\).

## Bound (sketch)

Let \(I_t^{\mathrm{fail}}=\mathbf{1}\{\mathrm{PI}_t>1\}\). Then
\[
\boxed{
\mathrm{Reg}_{\mathrm{SECDO}}
\;\le\;
\mathrm{Reg}_R
\;+\;
O\Bigl(\sum_t I_t^{\mathrm{fail}}\delta_t\Bigr).
}
\]

## Experiment protocol (Exp. 5)

- \(t\in[0,T_1)\): normal  
- \(t\in[T_1,T_2)\): corrupted \(\hat c\)  
- \(t\in[T_2,T]\): recovery  

Compare: SECDO (adaptive \(\alpha\)) vs fixed-\(\alpha\) vs reactive.
