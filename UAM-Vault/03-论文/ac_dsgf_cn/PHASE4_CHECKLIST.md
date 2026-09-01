# 第四阶段 · 工程化完成说明与检查清单

## 产物路径

```
paper/ac_dsgf_cn/
├── AC_DSGF_CN.md                 # Markdown 主稿（含图注与公式编号）
├── AC_DSGF_CN_导师版.docx        # 导师阅读版（章节+插图）
├── AC_DSGF_CN_PDF.pdf            # 图件审阅册（封面+Fig1–6）
├── EQUATIONS.md                  # 式 (1)–(6)
├── figures/Fig1–Fig6.png         # 300 dpi
├── tables/Table1–Table4.md
├── appendix/supplementary.md
└── ppt/AC_DSGF_导师汇报.pptx     # 10 页
```

## 再生命令

```bash
E:\ANACONDA\envs\dpg_hrl\python.exe scripts/gen_cn_paper_figures.py
E:\ANACONDA\envs\dpg_hrl\python.exe scripts/gen_cn_advisor_ppt.py
E:\ANACONDA\envs\dpg_hrl\python.exe scripts/export_cn_advisor_docx.py
E:\ANACONDA\envs\dpg_hrl\python.exe scripts/export_cn_review_pdf.py
E:\ANACONDA\envs\dpg_hrl\python.exe scripts/assemble_cn_paper.py
```

## 内容检查

- [x] Abstract / 摘要：通信约束动机 → 方法 → 可比性能 + Comm↓
- [x] Method 数学闭环：式 (1)–(5)
- [x] Table I 可信（冻结 CSV）
- [x] 无 causal utility
- [x] 无 Success outperform / superior / revolutionary / first-ever
- [x] 图 300 dpi；Caption 解释结论
- [x] PPT Future Work：ROS2/Gazebo · Hardware · Comm–Motion（无 ++）

## 给导师怎么发

1. 先发 **PPT**（10 页）  
2. 再发 **导师版.docx** 或 `AC_DSGF_CN.md`  
3. 英文 RA-L PDF 另附：`paper/ac_dsgf/AC_DSGF_v1.1.pdf`

## 评价（与你一致）

问题价值 / 工程意义 / 投稿匹配高；理论深度中等——定位为 **RA-L 机器人系统论文**，合适。
