# δ → Hausdorff for sum-budget sets (Thm1 bridge)

> **Canonical formal theorem:** [`thm1_scalar_budget_stability.md`](thm1_scalar_budget_stability.md)  
> This note remains a short pointer; cite Thm1 in papers.

**Feasible set**
\[
\mathcal{B}(c)
=
\{x\in\mathbb{R}^d:x\ge 0,\;\mathbf{1}^\top x\le c\},
\quad c\ge 0.
\]

## Claim

\[
d_H\bigl(\mathcal{B}(c),\mathcal{B}(\hat c)\bigr)
\le
\lvert c-\hat c\rvert.
\]

## Proof sketch

W.l.o.g. \(c\ge\hat c\). Then \(\mathcal{B}(\hat c)\subseteq\mathcal{B}(c)\), so
\[
\sup_{x\in\mathcal{B}(\hat c)}\mathrm{dist}\bigl(x,\mathcal{B}(c)\bigr)=0.
\]
For the other direction: take any \(x\in\mathcal{B}(c)\). If \(\mathbf{1}^\top x\le\hat c\), then \(x\in\mathcal{B}(\hat c)\).  
Otherwise let
\[
x'
=
\frac{\hat c}{\mathbf{1}^\top x}\,x
\quad(x\ge 0\Rightarrow x'\in\mathcal{B}(\hat c)).
\]
Then
\[
\|x-x'\|_2
=
\Bigl(1-\frac{\hat c}{\mathbf{1}^\top x}\Bigr)\|x\|_2
\le
\mathbf{1}^\top x-\hat c
\le
c-\hat c,
\]
using \(\|x\|_2\le\mathbf{1}^\top x\) for \(x\ge 0\) (since \(\|x\|_2\le\|x\|_1=\mathbf{1}^\top x\)).  
Hence
\[
\sup_{x\in\mathcal{B}(c)}\mathrm{dist}\bigl(x,\mathcal{B}(\hat c)\bigr)
\le
\lvert c-\hat c\rvert.
\]
Thus \(d_H\le\delta\) with \(\delta=\lvert c-\hat c\rvert\).

## Code logging

```text
delta_t = |c - c_hat|
dH_bound = 1.0 * delta_t
```

`models/secdo/theory/feasible_set_geometry.py`

## Use in Thm1 / Thm2

- Thm1: \(x=\Pi_{\mathcal{B}(\hat c)}(y)\in\mathcal{B}(\hat c)\); distance to true \(\mathcal{B}(c)\) controlled by \(\delta\).  
- Thm2: \(\sum\delta_t\) term is literally capacity forecast error, comparable to \(d_H\) error.
