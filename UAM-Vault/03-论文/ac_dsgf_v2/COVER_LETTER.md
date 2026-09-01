# Cover Letter Draft — SECDO-v2.0-submission

**Dear Editors / Area Chairs,**

We study **online optimization under evolving constraints**, where future feasibility information is only *partially predictable*.
We do **not** propose a novel neural optimizer for its own sake, nor a UAV engineering bake-off.

We propose **SECDO** (Self-Evolving Constrained Distributed Optimization), a learning-augmented framework whose scientific chain is:

\[
\textbf{constraint evolution}
\;\rightarrow\;
\textbf{predictability-conditioned projection}
\;\rightarrow\;
\textbf{dynamic regret upper bound}
\;(+\;\textbf{failure-safe degradation}).
\]

**Contributions (three):**
1. Formulate evolving feasible sets as a predictive optimization problem and relate set mismatch to capacity/state residuals (Thm.~1 + surrogate consistency).
2. Develop anticipatory / mixed projection with a dynamic regret *upper bound* governed by drift and set mismatch (Thm.~2).
3. Characterize when anticipation tightens *feasibility certificates* via the predictability index, and provide graceful degradation under prediction failure (Thm.~3, Cor.~4).

Empirically, synthetic experiments demonstrate regret *scaling structure*; UAV experiments demonstrate an optimality–safety trade-off; crash tests and ablations isolate failure containment.

We believe the manuscript fits venues interested in **online / constrained optimization** and **learning-augmented algorithms**, rather than deep RL or generic AI-agent tracks.

Sincerely,  
[Authors]

---

**一句话定位（投稿系统 “Summary” 可用）：**

> A learning-augmented online optimization framework for evolving constraints, where prediction quality determines when anticipatory optimization is beneficial and robustness mechanisms prevent degradation under prediction failures.
