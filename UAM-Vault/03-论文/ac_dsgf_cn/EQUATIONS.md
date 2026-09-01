# 公式编号一览（中文稿 · 与英文主稿对齐）

\[
\max_{\theta}
\mathbb{E}\!\left[\sum_{t} r_t\right]
-
\lambda_c\,
\mathbb{E}\!\left[\sum_{t} C_t\right]
\tag{1}
\]

\[
C_t=\sum_{i\neq j}g_{ij}^{t},\qquad
C=\frac1T\sum_{t}C_t
\tag{2}
\]

\[
g_{ij}^{t}
=
\sigma\!\bigl(W_g[h_i^{t};h_j^{t};d_{ij}^{t};\rho_{ij}^{t}]\bigr)
\cdot A_{ij}^{t}
\tag{3}
\]

\[
A_t^{\mathrm{AC}}=A^{t}\odot E_t^{K},\qquad
e_{ij}^{t}=A_{ij}^{\mathrm{AC}}\,q_{ij}^{t}\,g_{ij}^{t}
\tag{4}
\]

\[
a_i^{t}
\sim
\pi_{\theta}(\cdot\mid o_i^{t})
+\beta(t)\,\Delta(\Phi_i^{t})
\tag{5}
\]

\[
\mathrm{CEI}=\frac{S}{C+\varepsilon}
\tag{6}
\]
