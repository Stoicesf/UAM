# Reproducibility

**Version:** SECDO-v2.0-submission  
**GPU used in reported runs:** NVIDIA GeForce RTX 4060 Laptop GPU  
**Conda env:** `pytorch12`

## Environment (snapshot)

```yaml
python: "3.x (pytorch12 env)"
pytorch: "2.6+cu124 (env snapshot)"
cuda: "12.x (driver/runtime as installed)"
gpu: "RTX 4060 Laptop"
packages: [numpy, scipy, matplotlib, pyyaml]
```

Exact freeze: export with  
`conda env export -n pytorch12 > environment.yml`（可选，投稿后附上）。

## Training

```bash
# Predictor / SECDO stages (frozen configs under secdo/configs/train/)
python -m secdo.train mode=pretrain --config secdo/configs/train/pretrain.yaml
python -m secdo.train mode=joint --config secdo/configs/train/joint.yaml
python -m secdo.train mode=online --config secdo/configs/train/online.yaml
```

Checkpoints: `checkpoints/secdo_v2/{pretrain,joint,online}/`

## Evaluation

```bash
python scripts/run_all_secdo.py --exp all --methods secdo,reactive,oracle,dsgf,ac_dsgf --seeds 0 1 2 3 4
# or full pipeline resume
python scripts/run_secdo_v2_full.py --from uav --skip-data
python scripts/run_secdo_ablation.py
```

## Figures

```bash
python scripts/plot_secdo_v2_evidence_figures.py
python scripts/plot_secdo_v2_fig45.py
python scripts/verify_and_plot_final.py
powershell -File scripts/build_submission.ps1   # rebuild main.pdf
```

## QA

```bash
python scripts/audit_secdo_consistency.py
python scripts/final_claim_check.py
```
