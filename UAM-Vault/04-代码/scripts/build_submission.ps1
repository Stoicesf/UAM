# SECDO-v2.0-submission build chain (Windows)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "paper\ac_dsgf_v2")

Write-Host "=== SECDO Submission Build ==="
Write-Host "cwd: $(Get-Location)"

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if (-not $pdflatex) {
    Write-Host "ERROR: pdflatex not found. Install MiKTeX/TeX Live and ensure pdflatex is on PATH."
    exit 1
}

pdflatex -interaction=nonstopmode main.tex
if ((Test-Path "refs.bib") -and (Get-Command bibtex -ErrorAction SilentlyContinue)) {
    try { bibtex main } catch { }
}
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

Write-Host "=== PDF generated: paper/ac_dsgf_v2/main.pdf ==="
