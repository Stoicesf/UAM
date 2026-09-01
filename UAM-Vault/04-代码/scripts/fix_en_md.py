# -*- coding: utf-8 -*-
from pathlib import Path

p = Path(r"f:\UAM\paper\ac_dsgf\AC_DSGF_EN.md")
t = p.read_text(encoding="utf-8")

repls = [
    ("expensive.—", 'expensive.”'),
    ("high cost—.", 'high cost.”'),
    ("communication—.", 'communication”).'),
    ("always better.—", 'always better.”'),
    ("additional experiments—.", 'additional experiments”).'),
    ("### 4.1—.3 Three stages", "### 4.1–4.3 Three stages"),
    ("**On AC-random ≠AC-full Success.**", "**On AC-random ≥ AC-full Success.**"),
    ("Budget enters the learning loop (steps 2—),", "Budget enters the learning loop (steps 2–3),"),
    ("(~—?.57)", "(~−0.57)"),
    ("| **—?.4** (largest harm) |", "| **−3.4** (largest harm) |"),
    ("| drop bottom-\\(g\\) | 27.6% | ≠ |", "| drop bottom-\\(g\\) | 27.6% | +0.3 |"),
    (
        "| drop random | 31.0% | ≠ (no systematic harm) |",
        "| drop random | 31.0% | +3.7 (no systematic harm) |",
    ),
    (
        "| DSGF | × | × | —?| 21.7 | 40.29 |",
        "| DSGF | × | × | ✓ | 21.7 | 40.29 |",
    ),
    (
        "| AC-random (Random Drop) | random | × | —?| 28.5 | 20.32 |",
        "| AC-random (Random Drop) | random | × | ✓ | 28.5 | 20.32 |",
    ),
    (
        "| AC-no budget (\\(g{=}A\\)) | open | × | —?| 28.9 | 40.63 |",
        "| AC-no budget (\\(g{=}A\\)) | open | × | ✓ | 28.9 | 40.63 |",
    ),
    (
        "| **AC-full** | —?| —?| —?| 24.6 | **0.0077** |",
        "| **AC-full** | ✓ | ✓ | ✓ | 24.6 | **0.0077** |",
    ),
    (
        "| Full DSGF (\\(N{=}4\\)) | —| —| —?| 9.27 | —|",
        "| Full DSGF (\\(N{=}4\\)) | — | — | ✓ | 9.27 | — |",
    ),
    (
        "This is a conditional stability bound on actions—*not** a guarantee",
        "This is a conditional stability bound on actions—**not** a guarantee",
    ),
    ("E_t →Top-K", "E_t ← Top-K"),
    ("m_i →Aggregate", "m_i ← Aggregate"),
    ("a →π", "a ← π"),
    ("≠topology", "≠ topology"),
    ("≠who", "≠ who"),
    ("—Predefined", "— Predefined"),
    ("—Attention", "— Attention"),
    ("—Joint", "— Joint"),
    ("—Candidate", "— Candidate"),
    ("—Budget-Constrained", "— Budget-Constrained"),
    ("—Residual", "— Residual"),
    ("—Communication", "— Communication"),
    ("—Learned", "— Learned"),
    ("—Top-", "— Top-"),
    ("—Topology", "— Topology"),
    ("→Budget", "→ Budget"),
    ("→Residual", "→ Residual"),
    ("→topology", "→ topology"),
    ("→adaptive", "→ adaptive"),
    ("→residual", "→ residual"),
    ("→task-driven", "→ task-driven"),
]

for a, b in repls:
    t = t.replace(a, b)

p.write_text(t, encoding="utf-8")
print("EN cleaned", len(t))
