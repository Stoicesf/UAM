# Supplement S5 — Additional Exploration: Outcome-Aware Communication Learning
# RA-L optional · short · exploratory tone (NOT “failure analysis”)

---

\section*{S5.~Additional Exploration: Outcome-Aware Communication Learning}

\paragraph{Motivation.}
AC-DSGF learns \emph{adaptive communication topology} under a budget.
A natural extension is whether agents can further estimate
\emph{outcome-aware} message value (e.g., predicted influence on teammate
actions or returns) to refine who should talk.

\paragraph{Observation.}
Preliminary experiments indicate that long-horizon action-/outcome-influence
estimation remains challenging in cooperative navigation scenarios with sparse
rewards: utility alignment was unstable or collapsed, and we did not obtain a
main-claim-level Success–Comm improvement over AC-DSGF.

\begin{table}[h]
\centering
\caption{Indicative exploratory variants (not Table~I).}
\begin{tabular}{lcc}
\toprule
Variant & Cooperative outcome & Utility alignment \\
\midrule
AC-DSGF (learned gate) & Success hold, Comm↓ & --- \\
Action-influence utility & mixed & partial / unstable \\
Outcome-aware targets & degraded & collapsed / unstable \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Takeaway.}
We retain AC-DSGF’s learned topology and budget formulation as the primary
contribution, and leave long-horizon credit assignment for communication value
as future work toward communication–action co-evolution.
