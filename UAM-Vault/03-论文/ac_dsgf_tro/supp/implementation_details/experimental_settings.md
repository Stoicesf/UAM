# S3 — Experimental Settings (frozen)

## S3.1 Environment (primary formal setting)

| Item | Value |
|------|-------|
| Simulator | VMAS |
| Scenario | Cooperative navigation (T1) |
| Agents \(N\) (formal tables) | 16 |
| Scaling study \(N\) | \(\{16,32,64,128\}\) (§6.4; fixed \(K=4\)) |
| Parallel envs | 8 |
| Comm radius | 0.5 |
| Episode `max_steps` | 128 |
| Success threshold | 0.3 (config default where used) |

Config anchors: `configs/ac_dsgf/ac_dsgf_16uav.yaml`, `configs/scalability/uav16.yaml`.

## S3.2 Training hyperparameters (AC-DSGF 16-UAV formal)

| Item | Value |
|------|-------|
| Algorithm | guided MAPPO (`guided_mappo`) |
| Total frames | 102 400 (locked) |
| Eval episodes (train-time) | 200 |
| Seeds (formal evidence) | \(\{1234,2026,3407,42,8888\}\) |
| Hidden dim | 128 |
| Attention heads | 4 |
| Spatial layers | 2 |
| Guidance dim | 6 |
| Residual policy | true |
| \(\lambda_{\mathrm{comm}}\) | 0.001 |
| \(\beta\) schedule | warmup 0.2 · decay \(k=3\) · \(\beta\in[0,1]\) |

Checkpoint roots (frozen):  
`results/ac_dsgf/uav16/s{seed}` · baselines under `results/baseline16_seeds/...`

## S3.3 Communication settings (method-aware \(C\))

| Method | \(C(G_t)\) definition |
|--------|------------------------|
| MAPPO | \(0\) |
| GAT / DSGF | radius-neighborhood edge count |
| Full Attention | dense \(N(N-1)\) (at \(N=16\): \(240\)) |
| AC-DSGF | projected topology edge count after \(\Pi_{B_t}\) |

**Budgets / \(K\):** formal slices \(K\in\{2,4,6\}\) (§6.6); scaling fixed \(K=4\) (§6.4).

**Channel stress (§6.5; no retraining):**

| Factor | Levels |
|--------|--------|
| Packet loss \(p_\ell\) | \(\{0,0.1,0.3,0.5\}\) |
| Delay \(\tau\) | \(\{0,20,50,100\}\) ms → steps \(\{0,1,2,4\}\) |
| Bandwidth \(B/B_{\mathrm{full}}\) | \(\{0.1,0.2,0.4,0.8\}\) |

Protocols: `experiments/tro_6_*_formal_protocol.md`.
