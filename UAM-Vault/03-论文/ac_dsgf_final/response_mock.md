# Mock Review Response Skeleton (prepare before submission)

Use after v1.pdf. Answers stay **claim-safe**.

---

## Reviewer 1 — “Just attention / communication pruning?”

**Q:** Is AC-DSGF merely GAT + pruning / DSGF + penalty?

**A (use in rebuttal / Method IV-D):**

> No. Attention pruning typically removes messages from a **fixed** adjacency after (or independently of) policy learning. AC-DSGF formulates the communication topology as an **optimized decision variable** jointly learned with the policy:
> \(g_{ij}^{t}=\sigma(f(h_i,h_j,d,q))\), \(A_t^{\mathrm{AC}}=A\odot\mathrm{TopK}(g)\), under
> \(\max\mathbb{E}[\sum r_t]-\lambda_c\mathbb{E}[\sum C_t]\).
> Silence-collapse and trigger analyses show sparsity is task-driven, not random dropout.

---

## Reviewer 2 — “Limited performance gain”

**Q:** Success is not clearly higher than DSGF.

**A:**

> Our objective is **not** maximizing Success alone. The contribution is **maintaining task effectiveness under communication constraints**. At \(N=16\), AC-DSGF achieves Success comparable to DSGF (\(3.95\%\) vs \(4.01\%\)) with \(\sim 90\times\) lower communication cost and substantially higher CEI. Absolute Success remains low for all methods on this hard continuous benchmark; claims are comparative and communication-centric.

---

## Reviewer 3 — “Why not just shrink the communication radius?”

**Q:** A smaller \(R_c\) also reduces communication.

**A:**

> Radius reduction is **static** and geometry-only. AC-DSGF is **adaptive**: gates respond to latent state and interaction risk. Trigger analysis shows \(\mathrm{corr}(\mathrm{nn\_risk},C)\approx 0.84\) and negative correlation with dispersion; gate stability shows \(>99\%\) steps with \(\Delta E=0\) (non-chattering). Shrinking \(R_c\) cannot selectively open critical links when agents are close / high-risk while remaining silent otherwise.

---

## Extra expected asks

| Question | Short answer |
|----------|----------------|
| What is \(C=0.43\)? | Time-averaged soft gate mass \(\frac1T\sum_t\sum_{ij}g_{ij}\); baselines use edge-count intensity. |
| Soft ≠ packets? | Acknowledged in Limitations; we report intensity metrics consistently. |
| Packet loss? | Deferred to revision / RA-L extension (graceful degradation planned). |

---

## Preferred one-liner for any meta-question

> AC-DSGF optimizes **who/when/how much to communicate** under a budget, while residual guidance keeps communication assistive—not success chasing under unlimited messaging.
