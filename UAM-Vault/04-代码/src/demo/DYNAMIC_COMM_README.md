# Dynamic Communication Demo — paper narrative lock

## Safe claim
> AC-DSGF significantly reduces communication overhead while maintaining comparable task performance.

## Do NOT claim
> higher success by reducing communication.

Silence Collapse (16UAV eval) supports mechanism validation, not Success ranking:
- AC-full: S≈24.6%, Comm≈0.008
- AC-no budget: S≈28.9%, Comm≈40.6
→ ~three orders of magnitude Comm reduction, ~4pp Success trade-off.

## Demo role
Qualitative **Communication Emergence** (4 UAV viz):
GAT dense / DSGF dynamic / AC-DSGF sparse+task-triggered.

Formal numbers live in Table I++ / budget / ablation — not this video.

## Outputs
- `demo/videos/ac_dsgf_dynamic_comm.mp4`
- `demo/figures/communication_evolution.png`
- `demo/traces/*_episode.json`
