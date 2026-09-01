$N = Get-Content -LiteralPath "F:\UAM\scripts\vault_templates\names.json" -Encoding UTF8 -Raw | ConvertFrom-Json
$utf8 = New-Object System.Text.UTF8Encoding $false
$lines = @(
    ("home=" + $N.home),
    ("project=" + $N.sections.project),
    ("results=" + $N.sections.results),
    ("paper=" + $N.sections.paper),
    ("code=" + $N.sections.code),
    ("overview=" + $N.overview),
    ("freeze=" + $N.freeze)
)
[System.IO.File]::WriteAllLines("F:\UAM\scripts\vault_templates\_ps_names.txt", $lines, $utf8)
