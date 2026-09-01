# Build UAM-Vault: Obsidian knowledge base + delivery package.
# Usage (repo root): powershell -File scripts/build_uam_vault.ps1
# Chinese folder names come from vault_templates/names.json (UTF-8).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "INDEX.md"))) {
    $Root = (Get-Location).Path
}
$Vault = Join-Path $Root "UAM-Vault"
$Templates = Join-Path $PSScriptRoot "vault_templates"
$NamesPath = Join-Path $Templates "names.json"
$N = Get-Content -LiteralPath $NamesPath -Encoding UTF8 -Raw | ConvertFrom-Json

Write-Host "Root:  $Root"
Write-Host "Vault: $Vault"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Robo-Copy([string]$Src, [string]$Dst, [string[]]$ExtraArgs = @()) {
    if (-not (Test-Path -LiteralPath $Src)) {
        Write-Host "  skip (missing): $Src"
        return
    }
    Ensure-Dir $Dst
    $rcArgs = @($Src, $Dst, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/nc", "/ns", "/np") + $ExtraArgs
    & robocopy @rcArgs | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed: $Src -> $Dst (code $LASTEXITCODE)" }
    $global:LASTEXITCODE = 0
}

Ensure-Dir $Vault

# wipe previous vault contents (avoid leftover mojibake dirs from failed builds)
Get-ChildItem -LiteralPath $Vault -Force | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}

$SecProject = [string]$N.sections.project
$SecResults = [string]$N.sections.results
$SecPaper = [string]$N.sections.paper
$SecCode = [string]$N.sections.code
$Overview = [string]$N.overview
$FreezeName = [string]$N.freeze
$HomeName = [string]$N.home

foreach ($name in @($SecProject, $SecResults, $SecPaper, $SecCode)) {
    Ensure-Dir (Join-Path $Vault $name)
}

# --- templates ---
Copy-Item -LiteralPath (Join-Path $Templates "home.md") -Destination (Join-Path $Vault $HomeName) -Force
Copy-Item -LiteralPath (Join-Path $Templates "readme.md") -Destination (Join-Path $Vault "README.md") -Force
Copy-Item -LiteralPath (Join-Path $Templates "project.md") -Destination (Join-Path (Join-Path $Vault $SecProject) $Overview) -Force
Copy-Item -LiteralPath (Join-Path $Templates "results.md") -Destination (Join-Path (Join-Path $Vault $SecResults) $Overview) -Force
Copy-Item -LiteralPath (Join-Path $Templates "paper.md") -Destination (Join-Path (Join-Path $Vault $SecPaper) $Overview) -Force
Copy-Item -LiteralPath (Join-Path $Templates "code.md") -Destination (Join-Path (Join-Path $Vault $SecCode) $Overview) -Force

# --- .obsidian minimal ---
$Obs = Join-Path $Vault ".obsidian"
Ensure-Dir $Obs
Set-Content -LiteralPath (Join-Path $Obs "app.json") -Encoding UTF8 -Value @'
{
  "legacyEditor": false,
  "livePreview": true,
  "baseFontSize": 16,
  "showFrontmatter": true
}
'@
Set-Content -LiteralPath (Join-Path $Obs "core-plugins.json") -Encoding UTF8 -Value @'
{
  "fileExplorer": true,
  "globalSearch": true,
  "switcher": true,
  "graph": true,
  "backlink": true,
  "outgoingLink": true,
  "tagPane": true,
  "pagePreview": true,
  "dailyNotes": false,
  "templates": false,
  "noteComposer": true,
  "commandPalette": true,
  "editorStatus": true,
  "starred": true,
  "markdownImporter": false,
  "zkPrefixer": false,
  "randomNote": false,
  "outline": true,
  "wordCount": true,
  "slides": false,
  "audioRecorder": false,
  "workspaces": false,
  "fileRecovery": true,
  "publish": false,
  "sites": false,
  "sync": false,
  "canvas": true
}
'@
Set-Content -LiteralPath (Join-Path $Obs "appearance.json") -Encoding UTF8 -Value '{ "accentColor": "" }'

$ws = @{
    main = @{
        id = "root"
        type = "split"
        children = @(
            @{
                id = "leaf"
                type = "leaf"
                state = @{
                    type = "markdown"
                    state = @{ file = $HomeName; mode = "source" }
                }
            }
        )
        direction = "vertical"
    }
    left = @{
        id = "left"
        type = "side"
        children = @(
            @{ id = "files"; type = "leaf"; state = @{ type = "file-explorer"; state = @{} } }
        )
    }
    active = "leaf"
    lastOpenFiles = @($HomeName)
} | ConvertTo-Json -Depth 8
Set-Content -LiteralPath (Join-Path $Obs "workspace.json") -Encoding UTF8 -Value $ws

Write-Host "[1/4] project docs"
$Doc = Join-Path $Vault $SecProject
Copy-Item -LiteralPath (Join-Path $Root "README.md") -Destination (Join-Path $Doc "README.md") -Force
Copy-Item -LiteralPath (Join-Path $Root "INDEX.md") -Destination (Join-Path $Doc "INDEX.md") -Force
$relReadme = Join-Path $Root "release\README.md"
if (Test-Path -LiteralPath $relReadme) {
    Copy-Item -LiteralPath $relReadme -Destination (Join-Path $Doc "release-README.md") -Force
}
$FreezeDst = Join-Path $Doc $FreezeName
Robo-Copy (Join-Path $Root "paper\docs\freeze") $FreezeDst
$PlanSrc = Join-Path $Root "paper\docs\planning"
if (Test-Path -LiteralPath $PlanSrc) {
    Robo-Copy $PlanSrc (Join-Path $FreezeDst "planning")
}

Write-Host "[2/4] results (light)"
$Exp = Join-Path $Vault $SecResults
$resReadme = Join-Path $Root "results\README.md"
if (Test-Path -LiteralPath $resReadme) {
    Copy-Item -LiteralPath $resReadme -Destination (Join-Path $Exp "results-README.md") -Force
}
Robo-Copy (Join-Path $Root "paper\tables") (Join-Path $Exp "tables")
Robo-Copy (Join-Path $Root "paper\figures") (Join-Path $Exp "figures") @(
    "/XF", "*.pt", "*.pth", "*.mp4", "*.gif"
)

$RunsRoot = Join-Path $Exp "runs"
Ensure-Dir $RunsRoot
$KeepNames = @(
    "summary.json", "metrics.csv", "meta.json", "config.yaml", "TAG.txt",
    "action_alignment.csv", "communication.csv"
)
$ResultsSrc = Join-Path $Root "results"
if (Test-Path -LiteralPath $ResultsSrc) {
    Get-ChildItem -LiteralPath $ResultsSrc -Directory | ForEach-Object {
        $prefix = $_.Name
        $prefixDst = Join-Path $RunsRoot $prefix
        Ensure-Dir $prefixDst
        Get-ChildItem -LiteralPath $_.FullName -File -ErrorAction SilentlyContinue | Where-Object {
            ($KeepNames -contains $_.Name) -or ($_.Extension -in @(".csv", ".json", ".md", ".yaml", ".yml", ".txt"))
        } | ForEach-Object {
            Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $prefixDst $_.Name) -Force
        }
        Get-ChildItem -LiteralPath $_.FullName -Directory -ErrorAction SilentlyContinue | ForEach-Object {
            $runDst = Join-Path $prefixDst $_.Name
            Ensure-Dir $runDst
            foreach ($kn in $KeepNames) {
                $f = Join-Path $_.FullName $kn
                if (Test-Path -LiteralPath $f) {
                    Copy-Item -LiteralPath $f -Destination (Join-Path $runDst $kn) -Force
                }
            }
            $plots = Join-Path $_.FullName "plots"
            if (Test-Path -LiteralPath $plots) {
                Robo-Copy $plots (Join-Path $runDst "plots") @("/XF", "*.pt", "*.pth")
            }
        }
    }
}

Write-Host "[3/4] paper"
$PaperDst = Join-Path $Vault $SecPaper
Robo-Copy (Join-Path $Root "paper") $PaperDst @(
    "/XD", "__pycache__", ".git", "tensorboard",
    "/XF", "*.pt", "*.pth"
)
Copy-Item -LiteralPath (Join-Path $Templates "paper.md") -Destination (Join-Path $PaperDst $Overview) -Force

Write-Host "[4/4] code"
$Code = Join-Path $Vault $SecCode
$SrcOut = Join-Path $Code "src"
Ensure-Dir $SrcOut
$codeDirs = @(
    "algorithms", "env", "guidance", "models", "reward", "trainer",
    "tro", "secdo", "utils", "visualization", "experiments", "demo"
)
$xd = @("/XD", "__pycache__", ".git", "tensorboard", "checkpoints", "runs", "logs")
$xf = @("/XF", "*.pt", "*.pth", "*.pyc")
foreach ($d in $codeDirs) {
    $s = Join-Path $Root $d
    if (Test-Path -LiteralPath $s) {
        $extra = $xd + $xf
        if ($d -eq "demo") { $extra = $extra + @("/XF", "*.mp4", "*.gif", "*.webm") }
        Robo-Copy $s (Join-Path $SrcOut $d) $extra
    }
}
Robo-Copy (Join-Path $Root "configs") (Join-Path $Code "configs") ($xd + $xf)
Robo-Copy (Join-Path $Root "scripts") (Join-Path $Code "scripts") ($xd + $xf + @("/XD", "vault_templates"))
foreach ($f in @("train.py", "evaluate.py", "test_env.py", "requirements.txt", "run_train.bat")) {
    $fp = Join-Path $Root $f
    if (Test-Path -LiteralPath $fp) {
        Copy-Item -LiteralPath $fp -Destination (Join-Path $Code $f) -Force
    }
}
Copy-Item -LiteralPath (Join-Path $Templates "code.md") -Destination (Join-Path $Code $Overview) -Force

$allFiles = @(Get-ChildItem -LiteralPath $Vault -Recurse -File -ErrorAction SilentlyContinue)
$pt = @($allFiles | Where-Object { $_.Extension -in @(".pt", ".pth") })
$mb = [math]::Round((($allFiles | Measure-Object Length -Sum).Sum) / 1MB, 1)
Write-Host ""
Write-Host "Done. Vault size ~ $mb MB; files=$($allFiles.Count); .pt/.pth=$($pt.Count)"
if ($pt.Count -gt 0) {
    Write-Host "WARNING: weight files leaked:" -ForegroundColor Yellow
    $pt | Select-Object -First 10 | ForEach-Object { Write-Host "  $($_.FullName)" }
    exit 1
}

Get-ChildItem -LiteralPath $Vault | ForEach-Object { Write-Host ("  " + $_.Name) }
# section non-empty checks
foreach ($sec in @($SecProject, $SecResults, $SecPaper, $SecCode)) {
    $n = @(Get-ChildItem -LiteralPath (Join-Path $Vault $sec) -Recurse -File -EA SilentlyContinue).Count
    Write-Host ("  check $sec : $n files")
    if ($n -lt 2) { throw "section too empty: $sec" }
}
Write-Host "Open in Obsidian: $Vault"
