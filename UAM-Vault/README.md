# UAM-Vault

交付包 + Obsidian 本地知识库（由 `scripts/build_uam_vault.ps1` 生成）。

## 用 Obsidian 打开

1. 安装 [Obsidian](https://obsidian.md/)
2. 「打开文件夹作为库」→ 选择本目录（含 `.obsidian/` 的 `UAM-Vault`）
3. 从 `00-首页` 进入四区

## 对外交付

直接压缩整个 `UAM-Vault` 文件夹即可。已排除 `*.pt` / `tensorboard` 等重量级权重；完整实验 run 仍在原仓库 `results/`。

## 重新同步

在仓库根执行：

```powershell
powershell -File scripts/build_uam_vault.ps1
```

模板笔记在 `scripts/vault_templates/`；同步内容每次重建。

## 目录

| 分区 | 内容 |
|------|------|
| `01-项目说明` | README、INDEX、冻结文档 |
| `02-实验结果` | 轻量 run 摘要、表、图 |
| `03-论文` | `paper/` 全量 |
| `04-代码` | 源码与配置快照 |
