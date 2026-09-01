# Algorithm 1 — SECDO v2 (Final Freeze)

**Inference only.** Training: Stage I/II/online (separate).

```
Require: η, Enc, GRU, Dec, F_φ, F
Init: x_0 ∈ B(c_0), h_{-1}=0

For t = 0…T−1:
  // Step 1 — dynamics
  z_t ← Enc(s_t)
  h_t ← GRU(z_t, h_{t−1})
  ŝ_{t+1} ← Dec(h_t)

  // Step 2 — constraint dynamics
  ĉ_{t+1} ← F_φ(h_t)          // residual: c_t + Δ(h) OK
  // stop-grad of ĉ through Π in training

  // Step 3 — gradient step
  y_t ← x_t − η ∇F(x_t)

  // Step 4 — predictability
  δ̂_t ← |ĉ_{t+1} − c_t|       // online proxy (or tracked δ)
  χ̂_t ← |c_t − c_{t−1}|       // drift proxy (χ)
  PI_t ← δ̂_t / (χ̂_t + ε)
  α_t ← 1 / (1 + PI_t²)

  // Step 5 — mixed anticipatory projection
  c_mix ← α_t · ĉ_{t+1} + (1−α_t) · c_t
  x_{t+1} ← Π_{B(c_mix)}(y_t)

  Observe s_{t+1}, c_{t+1}; (optional) online L_c step with ||Δθ||≤ζ

Return x_T
```

| Mode | Projection budget |
|------|-------------------|
| Reactive | \(c_t\) |
| Oracle | \(c_{t+1}\) |
| SECDO | \(c^{\mathrm{mix}}(\alpha,\hat c,c_t)\) |
