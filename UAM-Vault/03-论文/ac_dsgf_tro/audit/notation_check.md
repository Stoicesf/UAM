# Notation Consistency Audit

## Locked symbols (main text)

| Symbol | Meaning | Forbidden |
|--------|---------|-----------|
| \(\phi_\theta\) | topology policy | \(\pi_\theta\) for the graph learner |
| \(\pi_\psi\) | physical action policy | bare \(\pi\) for topology |
| \(\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)\) | joint policy | ambiguous \(\Pi=(\pi,\phi)\) |
| \(\Pi_{B_t}\) | budget projection operator | “the optimal topology” |
| \(G_t=\phi_\theta(s_t)\) | decided graph (via projection in Alg.) | — |
| \(G_t^\star\) | **full-support reference topology before budget projection** | optimal / argmax / best graph |
| \(a_t=\pi_\psi(s_t,M(G_t))\) | action through message map | conflating \(a_t=\pi_\theta(\cdot)\) |
| \(C(G)\) / \(C_t\) | method-aware communication cost (§6.6) | mixing radius-edge count with dense \(N(N-1)\) inconsistently |

## Compact forms (allowed)

\[
a_t=\pi_\psi\bigl(s_t,\phi_\theta(s_t)\bigr)
\quad\text{(through \(M\))}
\]
is allowed **only** when the message map is stated nearby.

## Sweep results

| Check | EN | CN |
|-------|----|----|
| No \(\pi_\theta\) as topology | pass | pass |
| \(G^\star\) = pre-projection full-support reference | pass (§6.3) | pass |
| “optimal topology” only in prohibitions | pass | pass |
| Symbol freeze table present | pass (v0.11 adds \(G^\star\)) | pass |

## Residual

- Early MARL contrast uses \(a_t=\pi(s_t)\) for **standard MARL** (not our policy) — OK.
- Fig.~4 vs Fig.~7 both “communication scaling”: different sections (density \(\rho\) vs cost \(C_N\)); captions distinguish.
