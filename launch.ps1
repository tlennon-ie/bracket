# Bracket launcher (Windows PowerShell).
#
# Activates the repo venv, loads .env, and starts the unified server
# (FastAPI + static frontend) on http://127.0.0.1:8000.
#
# Flags pass through to bracket-serve.

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".venv"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[X] .venv not found. Run .\install.ps1 first." -ForegroundColor Red
    exit 1
}

# Load .env if present.
$envFile = Join-Path $RepoRoot ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.+)\s*$') {
            $name = $Matches[1]
            $value = $Matches[2]
            Set-Item -Path "env:$name" -Value $value
        }
    }
}

# Activate venv via direct python.exe invocation (avoids needing PS activation script).
$venvPython = Join-Path $VenvDir "Scripts\python.exe"
& $venvPython -m bracket.api.cli @args
