#!/usr/bin/env bash
# SECDO-v2.0-submission build chain
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/paper/ac_dsgf_v2"

echo "=== SECDO Submission Build ==="
echo "cwd: $(pwd)"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "ERROR: pdflatex not found. Install TeX Live / MiKTeX, or run scripts/build_submission.ps1 on Windows."
  exit 1
fi

pdflatex -interaction=nonstopmode main.tex
# bibliography optional (refs.bib may be absent)
if [[ -f refs.bib ]] && command -v bibtex >/dev/null 2>&1; then
  bibtex main || true
fi
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "=== PDF generated: paper/ac_dsgf_v2/main.pdf ==="
