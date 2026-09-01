# S2 — Topology Projection (no new algorithm)

**Main text:** Algorithm 1 · Theorem 1  
**Realization:** frozen AC-DSGF Top-\(K\) budget layer

## Inputs

| Input | Role |
|-------|------|
| \(s_t\) | swarm state (or local observations) |
| \(S_t=\phi_\theta(s_t)\) | edge scores from topology policy |
| \(A_t\) | candidate / geometric support (e.g. radius) |
| \(B_t\) or \(K\) | total budget or per-agent degree budget |

## Operator

\[
G_t=\Pi_{B_t}(S_t)
=
\arg\max_{G\in\mathcal{G}_{B_t}}
\sum_{(i,j)\in E(G)} g_{ij}.
\]

**Degree specialization (AC-DSGF):** for each agent \(i\), keep up to \(K\) highest-scoring admissible neighbors.

## Wording lock

- Call \(G_t\) a **budget-feasible utility-maximizing projection**.  
- Do **not** call it the task-optimal topology.  
- Soft \(\lambda\)-training ≠ hard feasibility; hard feasibility requires executing \(\Pi_{B_t}\).
