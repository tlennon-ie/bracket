# Bracket self-updater (Windows PowerShell).
#
# Pulls the latest changes from the configured git remote, re-runs the
# installer (idempotent -- picks up new Python/Node deps and rebuilds the
# frontend), and re-launches the server.
#
# Triggered by the React UI via POST /api/update/apply, which spawns this
# script detached from the dying API process. Output is appended to
# runs\update.log so the user can tail it after the restart.
#
# Re-run is safe; every step is idempotent.

$ErrorActionPreference = "Continue"
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "runs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
$LogPath = Join-Path $LogDir "update.log"

function Log($msg) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$stamp] $msg" | Out-File -FilePath $LogPath -Append -Encoding utf8
    Write-Host "[$stamp] $msg"
}

Log "==> Bracket update started"

# Give the API a moment to flush its response before we yank it.
Start-Sleep -Seconds 1

# Stop any running bracket-serve / launch processes on this repo. Best-effort.
try {
    $port = if ($env:BRACKET_PORT) { [int]$env:BRACKET_PORT } else { 8000 }
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        try {
            Log "Stopping process PID $($c.OwningProcess) holding port $port"
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
        } catch {
            Log "Could not stop PID $($c.OwningProcess): $_"
        }
    }
} catch {
    Log "Port scan failed (non-fatal): $_"
}

# Pull latest.
Log "==> git fetch + git pull"
& git fetch --tags 2>&1 | Tee-Object -FilePath $LogPath -Append | Out-Null
& git pull --ff-only 2>&1 | Tee-Object -FilePath $LogPath -Append | Out-Null
if ($LASTEXITCODE -ne 0) {
    Log "[X] git pull failed (exit $LASTEXITCODE). Aborting update -- your tree may have local changes."
    exit 1
}
Log "==> git submodule update --init --recursive"
& git submodule update --init --recursive 2>&1 | Tee-Object -FilePath $LogPath -Append | Out-Null
if ($LASTEXITCODE -ne 0) {
    Log "[X] git submodule update failed (exit $LASTEXITCODE)."
    exit 1
}

# Re-run the installer to pick up new deps and rebuild the frontend.
Log "==> Re-running install.ps1"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $RepoRoot "install.ps1") 2>&1 | Tee-Object -FilePath $LogPath -Append | Out-Null
if ($LASTEXITCODE -ne 0) {
    Log "[X] install.ps1 failed (exit $LASTEXITCODE). Server NOT relaunched. Check the log above."
    exit 1
}

# Relaunch.
Log "==> Relaunching server"
$launch = Join-Path $RepoRoot "launch.ps1"
if (-not (Test-Path $launch)) {
    Log "[X] launch.ps1 missing -- cannot relaunch. Run it manually."
    exit 1
}
# Windows PowerShell 5.1 mis-parses comma-separated -ArgumentList with
# line continuation (turns a trailing element into $null). Build the
# array explicitly via @() and pass via splatting to avoid the
# parameter-binding crash reported as "argument collection contains a
# null value".
$quotedLaunch = '"' + $launch + '"'
$launchArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $quotedLaunch)
$startParams = @{
    FilePath         = 'powershell.exe'
    ArgumentList     = $launchArgs
    WorkingDirectory = $RepoRoot
    WindowStyle      = 'Hidden'
}
try {
    Start-Process @startParams
    Log "==> Update complete. New server launching."
} catch {
    Log ("[X] Failed to relaunch server: " + $_)
    Log "    Run launch.ps1 manually."
    exit 1
}
