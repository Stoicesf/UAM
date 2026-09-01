# English RA-L sync status (post theory packaging)

**Date:** 2026-07-18

## Positioning (locked)
> Learning Adaptive Communication Topologies for Communication-Constrained UAV Swarm Coordination

Story: predefined topology → learn topology → sparse density scaling → deployable links.

## Synced files
| File | Status |
|------|--------|
| `ac_dsgf_main.tex` | title + 5-sentence abstract + keywords |
| `intro.tex` | topology-as-decision; 3 contributions |
| `method.tex` | A–D method + `\input{theory}` |
| `theory.tex` | **NEW** Prop.~2 $\eta_N$, lemmas, deployment |
| `experiments.tex` | Soft Mass; Sec.\ Scalability & Deployment (Fig.9–11) |
| `problem_definition.tex` | Soft Mass naming; drop 0.43 claim |
| `complexity.tex` | no “strictly linear” overclaim |
| `related_work.tex` | topology optimization wording |
| `COVER_LETTER.md` / `HIGHLIGHTS.md` / `CONTRIBUTION_STATEMENT.md` | aligned |
| `response_mock.md` | R1–R5 attack cards |

## Figures
`figures/Fig9_comm_density_scaling.png`, `Fig10_gate_distribution.png`, `Fig11_topk_deploy.png`, `framework.png`

## Do not
- Extend AC-DSGF++ / causal utility as main claims
- Advertise Top-$K$ Success gains as algorithm wins
- Claim Soft Mass = physical packets
- Claim full forward complexity is strictly $O(N)$

## Next (Step 3 submission pack)
- Compile PDF; fix any missing figure paths
- Optional: multi-seed Soft Mass re-eval under `step_mdp` (protocol honesty)
- Supplement + Code README freeze
