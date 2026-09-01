# S2.2 — Training Procedure (frozen backbone)

**No new modules / no unpublished tricks.**

## Joint object

\[
\Pi_{\theta,\psi}=(\pi_\psi,\phi_\theta)
\]

- \(\phi_\theta\): topology policy (edge utilities / scores)  
- \(\pi_\psi\): physical action policy (residual-corrected actor OK as in frozen AC-DSGF)

## Training surrogate

Maximize expected discounted task return with a soft communication cost surrogate \(\lambda c(G_t)\) (Lagrangian relaxation).  
Config anchor: `configs/ac_dsgf/ac_dsgf_16uav.yaml` (`lambda_comm: 0.001`, `algorithm: guided_mappo`).

## Evaluation

Hard projection \(\Pi_{B_t}\) (fixed-\(K\) or ratio budget) at rollout / evidence collection.  
Formal tables in the main paper use **frozen checkpoints** (no retraining for §6.2–6.6A evidence).
