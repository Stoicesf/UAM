# Mock Review Response Skeleton (Paper Hardening · claim-safe)

Aligned with `REVIEWER_MOCK_AND_CHECKLIST.md`. Use after `AC_DSGF_v1.1.pdf`.

---

## Reviewer 1 — “Just disabling / pruning / DSGF+ℓ₂?”

**Q:** Is communication reduction caused by simply disabling messages?

**A:**

> No. Random dropout deletes messages with probability $P(\mathrm{drop})$ on a fixed graph. Post-hoc pruning removes links after (or independently of) policy learning. AC-DSGF formulates topology as a **decision variable** co-optimized with control:
> \(g_{ij}^{t}=\sigma(W[h_i,h_j,d,\rho])\), \(A_t^{\mathrm{AC}}=A\odot\mathrm{TopK}(g)\), under
> \(\max\mathbb{E}[\sum r_t]-\lambda_c\mathbb{E}[\sum C_t]\).
> Silence ablation: forcing $g{=}A$ (full local broadcast) inflates Comm by over three orders of magnitude with only a small Success gain—showing that learned sparsity is not collapse. Gate stability (${>}99\%$ steps with $\Delta E_t{=}0$) and risk correlation ($0.84$) further show structured, non-random sparsity.

---

## Reviewer 2 — “Sparse communication hurts performance?”

**Q:** Success is not higher than DSGF / is performance sacrificed?

**A:**

> Our objective is **not** maximizing Success alone. At $N{=}16$ (5 seeds), AC-DSGF achieves Success **comparable** to DSGF ($3.95\%\pm0.83$ vs $4.01\%\pm1.68$) while reducing communication from $39.27$ to $0.43$ ($\sim$91$\times$) and raising CEI by nearly $90\times$. Absolute Success remains low for **all** methods on this hard continuous benchmark; claims are comparative and communication-centric. Budget sweeps show graceful degradation under tighter $\rho$.

---

## Reviewer 3 — “Scalability / smaller radius enough?”

**Q:** Why not shrink $R_c$? Does this scale?

**A:**

> A smaller radius is **static and geometry-only**. AC-DSGF is **adaptive**: communication intensifies under proximity risk and attenuates when agents disperse ($\mathrm{corr}(\mathrm{risk},C)\!\approx\!0.84$). Complexity-wise, Top-$K$ budgeting yields attended support $\mathcal{O}(NK)$ versus dense $\mathcal{O}(N^{2})$ (Lemma~1). Soft mass $C=\sum g_{ij}$ is an intensity proxy; mapping to physical packets is discussed as a limitation, with packet-loss diagnostics showing flat Success under random edge drops.

---

## Preferred meta one-liner

> AC-DSGF learns **who/when/how much** to communicate under a budget, while residual guidance keeps communication assistive—not Success chasing under unlimited messaging.
