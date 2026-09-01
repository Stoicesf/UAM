# SECDO Research Platform (v2 Final Freeze)

Independent of frozen T-RO (`paper/ac_dsgf_tro/`).  
Theory: `paper/ac_dsgf_v2/theory/FINAL_THEORY_FREEZE.md`  
Algorithm: `paper/ac_dsgf_v2/algorithm_secdo_v2.md`

## Statement

SECDO is **not** “prediction + projected gradient”.  
It is a learning-driven framework that identifies, anticipates, and optimizes under evolving feasible regions
\[
\mathcal{B}_t \to \hat{\mathcal{B}}_{t+1} \to \Pi \to x_{t+1}
\]
with predictability index \(\mathrm{PI}=\delta/\chi\) and mixed budget \(c^{\mathrm{mix}}=\alpha\hat c+(1-\alpha)c\).

## Train

```bash
python -m secdo.train mode=pretrain --config secdo/configs/train/pretrain.yaml
python -m secdo.train mode=joint --config secdo/configs/train/joint.yaml
python -m secdo.train mode=online --config secdo/configs/train/online.yaml
```

## Experiments

```bash
python -m secdo.experiments.synthetic_convex.run
python -m secdo.experiments.pi_boundary.run
python -m secdo.experiments.crash_recovery.run
python scripts/run_all_secdo.py --exp fast_drift --methods secdo,reactive,oracle --seeds 0 1 2
```

## Layout

```text
secdo/
  datasets/uav/     # canonical teacher c_t
  models/           # SECDO + Predictor + CapacityHead
  optimizer/        # PI/α/c_mix + Π
  training/         # unified Trainer
  baselines/        # get_solver(...).solve(env)
  evaluation/       # regret, violation, theory metrics, plots
  experiments/      # synthetic / UAV / PI / crash
```
