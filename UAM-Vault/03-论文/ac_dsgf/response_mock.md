# Response Mock — AC-DSGF v1（RA-L 投稿防御）

**定位一句：** We treat communication topology as a first-class decision variable under soft budgets, learning *when and with whom* to communicate. Our advantage is **SCA**—task-effective coordination at near-zero activation density—not brute-force Success.

---

## Q1 — Why not train with hard Top-K from the start?

**Attack:** Soft gates are unnecessary; just optimize discrete Top-K.

**Response:**
1. Soft gates \(g_{ij}\in[0,1]\) enable differentiable exploration under \(\mathbb{E}[R-\lambda_c C]\).
2. Hard Top-K is a **deployment approximation** of the learned soft pattern (Fig. 9), not a replacement training objective.
3. Under matched hard \(K\), AC is **comparable** to Distance/Random (Table 2b / Fig. 5); the primary gain appears in the **soft SCA regime** (Table 1), where activation is orders of magnitude lower than dense baselines.
4. We do **not** claim Top-K training would be inferior; we claim soft training + hard deployment is a practical pipeline.

---

## Q2 — Why is AC-random Success higher than AC-full (Table 2)?

**Attack:** If random gates get 28.5% vs AC-full 24.6%, why learn?

**Response:**
1. **Magnitude gap:** AC-random SCA is **~2,600×** larger (20.32 vs 0.0077). Higher Success without a budget is not the objective.
2. Under a **hard budget** (≤\(K\) edges/agent), random selection is not a feasible scheduler guarantee; AC-full’s sparse topology is deployable as-is.
3. Edge-importance ablation (Fig. 10): dropping top-\(g\) edges hurts tasks; dropping bottom/random does not → learned ranking is task-aligned, not chance.
4. Thesis sentence: advantage = **comparable Success at near-zero activation density**, the prerequisite for bandwidth-limited swarms.

---

## Q3 — Prop.1 is only an upper bound; is it tight?

**Attack:** \(\eta_N\le K/(N-1)\) is trivial degree counting.

**Response:**
1. Prop.1 is intentionally a **degree-constraint scaling interpretation**, not a learning-convergence theorem and not a claim of equality \(|E_t|=KN\).
2. Empirically, AC maintains low normalized density as \(N\) grows (Fig. 7), consistent with \(\mathcal{O}(1/N)\) decay versus denser baselines.
3. Tightness would require characterizing realized degree vs \(K\); we report measured SCA/\(\eta_N\) rather than claiming a tight analytic constant.
4. Prop.2/3 remain interpretive (action stability; induced budgeted selection)—aligned with RA-L evidence standards.

---

## Quick extras (if raised)

| Attack | One-liner |
|--------|-----------|
| Soft Mass ≠ packets | SCA is a training proxy; deployment uses threshold / Top-K hard edges. |
| Attention = topology | Attention weights messages on a support; gates decide activation under budget. |
| Need multi-λ Pareto | Fig. 11 is a frozen-policy operating curve; multi-λ retrain is optional appendix, not claimed in main text. |
| SwarmOS / PX4 contribution | Out of scope; paper contribution is algorithmic topology learning. |
