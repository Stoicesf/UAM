# UAM-Vault

> 交付包 + Obsidian 本地知识库。源仓库路径不变；本目录由 `scripts/build_uam_vault.ps1` 同步生成。

## 主线（冻结）

**AC-DSGF v1**：可学习通信拓扑 + 预算优化 + residual guidance（Success≈DSGF，Comm↓~90×）。  
AC-DSGF++ = Supplement 负结果；T-RO / SECDO-v2 为并行线，勿并入 v1 主结论。

## 四区入口

- [[01-项目说明/总览|01 项目说明]]
- [[02-实验结果/总览|02 实验结果]]
- [[03-论文/总览|03 论文]]
- [[04-代码/总览|04 代码]]

## 快速跳转

| 需求 | 位置 |
|------|------|
| 仓库 README | [[01-项目说明/README]] |
| 项目 INDEX | [[01-项目说明/INDEX]] |
| 最终冻结 | [[01-项目说明/冻结与路线图/AC_DSGF_FINAL_FREEZE]] |
| 结果表 | [[02-实验结果/tables]] |
| T-RO 稿 | [[03-论文/ac_dsgf_tro/README]] |
| 代码地图 | [[04-代码/总览]] |

## 使用

1. Obsidian →「打开文件夹作为库」→ 选本目录 `UAM-Vault`
2. 对外交付：压缩整个 `UAM-Vault`（已排除 checkpoint）
3. 重新同步：在仓库根执行 `powershell -File scripts/build_uam_vault.ps1`
