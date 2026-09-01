# SECDO-v2.0-submission Package

**工作树：** `paper/ac_dsgf_v2/`（勿改 `paper/ac_dsgf_tro/`）  
**干净投稿快照：** `paper/SECDO-v2.0-submission/`  
**版本：** [`VERSION`](VERSION) = `SECDO-v2.0-submission`  
**红队结论：** [`REDTEAM_REVIEW.md`](REDTEAM_REVIEW.md) — **冻结，进入投稿**

## 投稿树

```
SECDO-v2.0-submission/
├── main.pdf
├── main.tex
├── sections/
├── appendix/          # *.tex only
├── figures/
├── supplementary/
├── COVER_LETTER.md
├── rebuttal_prepare_final.md
├── reviewer_attack/
├── README.md
└── VERSION
```

## 编译

```bash
powershell -File scripts/build_submission.ps1
# or from paper/ac_dsgf_v2:
pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

## Claim 扫描

```bash
python scripts/final_claim_check.py
```

## 叙事锁（不可拆）

```
constraint evolution → anticipatory/mixed projection → PI-conditioned α → dynamic regret + recovery
```

**禁止：** Phase 7 / 新定理 / 新实验 / claim 漂移。
