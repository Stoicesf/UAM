# SECDO-v2.0-submission Release Manifest

**Do not invent `v2.1` / `v2-final` without a post-review cycle.**  
**Tag name:** `SECDO-v2.0-submission`  
**定位：** learning-augmented online optimization for *evolving constraints*（不是 “better UAV optimizer”）。

| Bundle (logical) | Source |
|------------------|--------|
| `paper/` | `paper/ac_dsgf_v2/` (`main.tex`, `main.pdf`, figures, appendix, supplementary) |
| `code/` | `secdo/` |
| `configs/` | `secdo/configs/` |
| `results/` | `results/secdo_v2/` |
| `scripts/` | `scripts/build_submission.*`, audit / claim / plot |

---

## Environment

```yaml
python: 3.x
conda_env: pytorch12
pytorch: 2.6+cu124
cuda: 12.x
gpu: RTX 4060 Laptop
```

Optional: `conda env export -n pytorch12 > environment.yml`

---

## Training

```bash
python -m secdo.train mode=pretrain --config secdo/configs/train/pretrain.yaml
python -m secdo.train mode=joint --config secdo/configs/train/joint.yaml
python -m secdo.train mode=online --config secdo/configs/train/online.yaml
```

---

## Evaluation

```bash
python scripts/run_all_secdo.py --exp all --methods secdo,reactive,oracle --seeds 0 1 2
# full seed suite used in paper: 0 1 2 3 4
```

---

## Figures & PDF

```bash
python scripts/verify_and_plot_final.py
powershell -File scripts/build_submission.ps1   # → paper/ac_dsgf_v2/main.pdf
```

---

## Pre-submission QA

```bash
python scripts/audit_secdo_consistency.py
python scripts/final_claim_check.py
```

---

## Git tag（若仓库已初始化）

```bash
git tag -a SECDO-v2.0-submission -m "Submission frozen version"
git push origin SECDO-v2.0-submission
```

> 当前工作区 `f:\UAM` **尚无 `.git`**；请在正式仓库根目录执行上述命令。冻结内容以本 manifest + `paper/ac_dsgf_v2/VERSION` 为准。

---

## Freeze docs

- `paper/ac_dsgf_v2/PHASE6_SUBMISSION_RELEASE.md`
- `paper/ac_dsgf_v2/COVER_LETTER.md`
- `paper/ac_dsgf_v2/PDF_PAGE_CHECKLIST.md`
- `paper/ac_dsgf_v2/supplementary/`
- `paper/rebuttal_prepare_final.md`
- `paper/final_review_simulation.md`
