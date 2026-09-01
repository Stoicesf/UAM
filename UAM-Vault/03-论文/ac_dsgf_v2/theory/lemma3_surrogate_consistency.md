# Lemma 3 — Surrogate Consistency

**Status:** FROZEN (SECDO v2)

## Statement

Let the capacity teacher \(c=T(s)\) be \(L_g\)-Lipschitz in the feature/state used by \(\mathcal{F}_\phi\).  
If the training loss includes \(L_c=\|c-\hat c\|^2\), then (pointwise / in expectation)
\[
\boxed{
\delta_t
\;=\;
\lvert c_t-\hat c_t\rvert
\;\le\;
L_g\sqrt{L_c}
}
\]
up to absolute constants from norm equivalences. Therefore minimizing \(L_c\) directly reduces the \(\delta\) term appearing in Thm1–Thm2.

## Training implication

Stage-I weights \(\lambda_c>\lambda_s>\lambda_u\) prioritize the theory-relevant surrogate.
