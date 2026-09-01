# Theorem 3 — Anticipatory Projection Advantage

**Status:** FROZEN (SECDO v2)  
**Depends on:** Thm1, Lemma2 / anticipatory certificates

## Statement

Let
\[
\chi_t=d_H(\mathcal{B}_{t+1},\mathcal{B}_t),\qquad
\delta_t=d_H(\hat{\mathcal{B}}_{t+1},\mathcal{B}_{t+1}),\qquad
\mathrm{PI}_t=\frac{\delta_t}{\chi_t+\varepsilon}.
\]
Reactive / anticipatory violation certificates (scalar budget):
\[
V_R(t)\le\chi_t,\qquad V_A(t)\le\delta_t.
\]
If \(\mathrm{PI}_t<1\) (\(\delta_t<\chi_t\)), then \(V_A(t)<V_R(t)\) in upper bound:
\[
\boxed{
\delta_t<\chi_t
\;\Rightarrow\;
\text{anticipatory projection has the tighter violation certificate}.
}
\]

## Remark

Advantage is **conditional**, not universal. When \(\mathrm{PI}_t>1\), Corollary 4 shrinks \(\alpha_t\) toward reactive.
