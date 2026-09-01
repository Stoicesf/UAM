# Supplementary Material（SECDO-v2.0-submission）

主文 `main.pdf` 为投稿正文；本目录为审稿可附材料索引（内容指向已冻结附录，**不复制改写定理**）。

| 文件 | 内容 |
|------|------|
| `reproducibility.md` | 环境、训练、评估、出图命令 |
| `hyperparameters.yaml` | 训练/损失/数据关键超参快照 |
| `proof_details.md` | → 主文 Appendix A–E + `appendix/B_proofs.md` |
| `implementation_details.md` | → `appendix/implementation.tex` + `C_algorithm_details.md` |
| `additional_results.md` | → ablation / PI boundary / crash 路径索引 |

上传审稿系统时，可将上述 Markdown 与 `../main.pdf` 中 appendix页一并打包为 `supplementary.zip`；若编辑部要求 PDF-only，用现有 `main.pdf` 附录页即可（Complexity / proofs 已在主 PDF）。
