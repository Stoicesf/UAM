# -*- coding: utf-8 -*-
"""Assemble AC_DSGF_CN.md from chapter files (Phase 4)."""
from pathlib import Path

CN = Path(__file__).resolve().parents[1] / "paper" / "ac_dsgf_cn"

HEADER = """# 面向通信受限无人机蜂群协同的自适应通信拓扑学习方法

> **英文对应：** Adaptive Communication-Constrained DSGF for Efficient UAV Swarm Coordination  
> **状态：** 第四阶段工程化完成（图件 · 公式编号 · 导师 PPT/DOCX）  
> **主张锁定：** 任务性能保持 + 通信约降低两个数量级 + 拓扑自主优化  
> **非声称：** Success 超越；因果效用 / AC-DSGF++

**作者：** ［待填］  
**用途：** 导师审阅 / 开题 / 项目申请 / 中文论文初稿

**配套产物：** `AC_DSGF_CN_导师版.docx` · `ppt/AC_DSGF_导师汇报.pptx` · `figures/Fig1–Fig6.png`

---

"""


def load(name: str) -> str:
    t = (CN / name).read_text(encoding="utf-8")
    if name == "01_摘要.md":
        t = t.replace("# 摘要", "## 摘要", 1)
    if name == "02_引言.md":
        import re
        t = re.sub(
            r"(?s)## 1\.6 文章结构.*$",
            "## 1.6 文章结构\n\n"
            "- 第 2–4 章：相关工作 · 问题定义 · 方法（式 (1)–(5)）\n"
            "- 第 5–6 章：实验 · 复杂度（Fig.4–6）\n"
            "- 第 7–8 章：讨论 · 结论\n",
            t,
        )
    return t.strip() + "\n"


def main():
    parts = [
        HEADER,
        load("01_摘要.md"),
        "\n---\n",
        load("02_引言.md"),
        "\n---\n",
        load("03_相关工作.md"),
        "\n---\n",
        load("04_问题定义.md"),
        "\n---\n",
        load("05_方法.md"),
        "\n---\n",
        load("06_实验.md"),
        "\n---\n",
        load("07_复杂度分析.md"),
        "\n---\n",
        load("08_讨论.md"),
        "\n---\n",
        load("09_结论.md"),
        "\n---\n",
        load("EQUATIONS.md"),
        "\n---\n",
        load("appendix/supplementary.md"),
        """
---

## 导师审阅检查清单

- [x] 无 causal utility / AC-DSGF++ 主叙事
- [x] 无 Success outperform / superior
- [x] 表 1 绑定主声称；诊断实验分开
- [x] Fig.1–Fig.6 已生成（300 dpi）
- [x] 公式 (1)–(6) 统一编号
- [x] 导师 PPT 10 页
- [ ] 人工通读中文润色（建议组会前）
- [ ] Word 中公式渲染（必要时用 MathType / OMML）
""",
    ]
    out = CN / "AC_DSGF_CN.md"
    out.write_text("".join(parts), encoding="utf-8")
    print("wrote", out, "bytes", out.stat().st_size)


if __name__ == "__main__":
    main()
