"""Day 10: scan SECDO submission prose for over-claims.

Checks paper/ac_dsgf_v2 LaTeX + key drafts.
Negations / redline quotes are allowed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Primary submission artifacts (LaTeX is authoritative for Day 8–10)
SCAN_GLOBS = [
    ROOT / "paper" / "ac_dsgf_v2" / "sections" / "*.tex",
    ROOT / "paper" / "ac_dsgf_v2" / "appendix" / "proof_*.tex",
    ROOT / "paper" / "ac_dsgf_v2" / "appendix" / "additional_exp.tex",
    ROOT / "paper" / "ac_dsgf_v2" / "appendix" / "implementation.tex",
    ROOT / "paper" / "ac_dsgf_v2" / "main.tex",
    ROOT / "paper" / "ac_dsgf_v2" / "drafts" / "SECDO_INTRO_RESULTS_CONCLUSION.md",
    ROOT / "paper" / "rebuttal_prepare_final.md",
]

FORBIDDEN = [
    (r"guarantee[sd]?\s+optimal", "absolute optimality"),
    (r"achieves?\s+the\s+theoretical\s+regret\s+bound", "fitted-bound overclaim"),
    (r"achieves?\s+optimal\s+regret", "optimal regret"),
    (r"secdo\s+always\s+wins", "universal win"),
    (r"always\s+outperforms", "universal win"),
    (r"prediction-enhanced\s+projected", "downgrade narrative"),
    (r"prediction-enhanced\s+pgd", "downgrade narrative"),
    (r"fully\s+recovers", "overclaim recovery"),
    (r"learns?\s+the\s+true\s+future\s+constraint", "oracle learning claim"),
    (r"\bverif(?:y|ies|ying)\s+theorem", "UAV≠theory verify"),
    (r"pareto\s+optimal", "pareto claim"),
]

NEG = re.compile(
    r"("
    r"\bnot\b|n't\b|never\b|without\b|rather than\b|instead of\b|"
    r"do not\b|does not\b|did not\b|cannot\b|can not\b|"
    r"禁止|不要写|并非|不是|非\s*|避免|红线|❌|"
    r"no claim|not claimed|not a claim|we do not|does not claim"
    r")",
    re.I,
)


def _strip_tex(s: str) -> str:
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"``|''", '"', s)
    s = re.sub(r"\\+", " ", s)
    return s


def _targets():
    for g in SCAN_GLOBS:
        yield from sorted(g.parent.glob(g.name)) if g.name.startswith("*") or "*" in g.name else (
            [g] if g.is_file() else []
        )


def main() -> int:
    # fix glob handling
    paths: list[Path] = []
    base = ROOT / "paper" / "ac_dsgf_v2"
    paths.extend(sorted((base / "sections").glob("*.tex")))
    for name in (
        "proof_thm1.tex",
        "proof_surrogate.tex",
        "proof_lemma2.tex",
        "proof_thm2.tex",
        "proof_thm3.tex",
        "proof_cor4.tex",
        "additional_exp.tex",
        "implementation.tex",
    ):
        p = base / "appendix" / name
        if p.is_file():
            paths.append(p)
    for p in (base / "main.tex", base / "SECDO_PAPER_ZH.md", ROOT / "paper" / "rebuttal_prepare_final.md"):
        if p.is_file():
            paths.append(p)

    hits = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("%"):
                continue
            plain = _strip_tex(line)
            low = plain.lower()
            # redline quote lists / hygiene bullets
            if NEG.search(low) or low.strip().startswith("- \"") or low.strip().startswith("- “"):
                continue
            if "禁止" in plain or "红线" in plain:
                continue
            for pat, tag in FORBIDDEN:
                if re.search(pat, low):
                    hits.append((str(path.relative_to(ROOT)), i, tag, plain.strip()[:140]))

    out = base / "FINAL_CLAIM_CHECK.md"
    lines = ["# Final Claim Check", "", "Scope: submission `.tex` + locked draft/rebuttal.", ""]
    if not hits:
        lines += ["**Result: PASS**", ""]
        ok = True
    else:
        lines += ["**Result: FAIL**", "", "| File | Line | Tag | Text |", "|------|------|-----|------|"]
        for f, i, tag, t in hits:
            lines.append(f"| `{f}` | {i} | {tag} | {t.replace('|', '/')} |")
        lines.append("")
        ok = False
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
