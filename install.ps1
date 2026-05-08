# Bracket installer (Windows PowerShell).
#
# What it does:
#   1. Verifies Python 3.10+ is available.
#   2. Creates a .venv next to the project.
#   3. Detects GPU (nvidia-smi) and selects a matching PyTorch wheel.
#   4. Installs Bracket + its dependencies.
#   5. Clones the trainers Bracket drives (sd-scripts, musubi-tuner) into
#      $env:LOCALAPPDATA\bracket\trainers and creates a shared trainer venv.
#   6. Writes a .env with sensible defaults so the UI lands ready to run.
#
# Re-run is safe -- every step is idempotent.

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$TrainersRoot = if ($env:BRACKET_TRAINERS_ROOT) { $env:BRACKET_TRAINERS_ROOT } else { Join-Path $env:LOCALAPPDATA "bracket\trainers" }
$VenvDir = Join-Path $RepoRoot ".venv"
$TrainerVenv = Join-Path $TrainersRoot "venv"

function Write-Step($msg)  { Write-Host "==> $msg" -ForegroundColor Blue }
function Write-Ok($msg)    { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[!]  $msg" -ForegroundColor Yellow }
function Write-FailExit($msg) {
    Write-Host "[X] $msg" -ForegroundColor Red
    exit 1
}

# --- 1. Python ---------------------------------------------------------
Write-Step "Checking Python"
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $python) {
    Write-FailExit "Python not found. Install Python 3.10+ from https://www.python.org/"
}
$pyVersion = & $python.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pyMajor = [int]($pyVersion.Split('.')[0])
$pyMinor = [int]($pyVersion.Split('.')[1])
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 10)) {
    Write-FailExit "Python $pyVersion found, but Bracket needs >=3.10."
}
Write-Ok "Python $pyVersion"

# --- 2. Hardware -------------------------------------------------------
Write-Step "Detecting hardware"
$torchIndex = ""
$gpuDesc = "CPU only"
$cudaTag = "cpu"
$nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvsmi) {
    $gpuName = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    $driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($gpuName) {
        if ($driver) {
            $driverMajor = [int]($driver.Split('.')[0])
            if ($driverMajor -ge 555) { $torchIndex = "https://download.pytorch.org/whl/cu124"; $cudaTag = "cu124" }
            elseif ($driverMajor -ge 525) { $torchIndex = "https://download.pytorch.org/whl/cu121"; $cudaTag = "cu121" }
            else { $torchIndex = "https://download.pytorch.org/whl/cu118"; $cudaTag = "cu118" }
        }
        $gpuDesc = "$gpuName (driver $driver -> $cudaTag)"
    }
}
Write-Ok "GPU: $gpuDesc"

# --- 3. Repo venv ------------------------------------------------------
Write-Step "Creating repo venv at .venv\"
if (-not (Test-Path $VenvDir)) {
    & $python.Source -m venv $VenvDir
}
$venvPython = Join-Path $VenvDir "Scripts\python.exe"
& $venvPython -m pip install --quiet --upgrade pip wheel
Write-Ok "Repo venv ready"

# --- 4. Bracket install -----------------------------------------------
Write-Step "Installing Bracket (editable + dev)"
& $venvPython -m pip install --quiet -e ".[dev]"
Write-Ok "Bracket installed"

# --- 5. Trainers + trainer venv ---------------------------------------
Write-Step "Setting up trainers at $TrainersRoot"
New-Item -ItemType Directory -Force -Path $TrainersRoot | Out-Null
$musubiDir = Join-Path $TrainersRoot "musubi-tuner"
$sdScriptsDir = Join-Path $musubiDir "sd-scripts"

if (-not (Test-Path $musubiDir)) {
    git clone --depth=1 https://github.com/kohya-ss/musubi-tuner $musubiDir
    Write-Ok "Cloned musubi-tuner"
} else {
    Write-Ok "musubi-tuner already present"
}
if (-not (Test-Path $sdScriptsDir)) {
    git clone --depth=1 https://github.com/kohya-ss/sd-scripts $sdScriptsDir
    Write-Ok "Cloned sd-scripts"
} else {
    Write-Ok "sd-scripts already present"
}

if (-not (Test-Path $TrainerVenv)) {
    & $python.Source -m venv $TrainerVenv
}
$trainerPython = Join-Path $TrainerVenv "Scripts\python.exe"
& $trainerPython -m pip install --quiet --upgrade pip wheel

Write-Step "Installing PyTorch into trainer venv ($gpuDesc)"
if ($torchIndex) {
    & $trainerPython -m pip install --quiet torch torchvision --index-url $torchIndex
} else {
    & $trainerPython -m pip install --quiet torch torchvision
    Write-Warn "No NVIDIA GPU detected -- installed CPU-only PyTorch. Training will be very slow."
}
Write-Ok "PyTorch installed"

Write-Step "Installing trainer dependencies"
$musubiReq = Join-Path $musubiDir "requirements.txt"
$sdReq = Join-Path $sdScriptsDir "requirements.txt"
if (Test-Path $musubiReq) { & $trainerPython -m pip install --quiet -r $musubiReq }
if (Test-Path $sdReq)     { & $trainerPython -m pip install --quiet -r $sdReq }
Write-Ok "Trainer deps installed"

# --- 6. .env defaults --------------------------------------------------
Write-Step "Writing .env defaults"
$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    $envBody = @"
# Bracket -- environment defaults. Generated by install.ps1.
# Override anything below by editing this file or setting in your shell.

BRACKET_TRAINERS_ROOT=$TrainersRoot
BRACKET_VENV_PYTHON=$trainerPython
BRACKET_MUSUBI_DIR=$musubiDir
BRACKET_SD_SCRIPTS_DIR=$sdScriptsDir

# Add when you have weights downloaded:
# BRACKET_VAE_PATH=C:\path\to\ae.safetensors
# BRACKET_QWEN3_TE_PATH=C:\path\to\qwen_3_4b.safetensors
# BRACKET_FLUX2_DIT_PATH=C:\path\to\flux-2-klein-base-9b-fp8.safetensors
# BRACKET_MISTRAL3_TE_PATH=C:\path\to\mistral_3_small_flux2_fp8.safetensors
"@
    Set-Content -Path $envFile -Value $envBody -Encoding utf8
    Write-Ok "Wrote .env"
} else {
    Write-Ok ".env already exists -- leaving alone"
}

Write-Host ""
Write-Ok "Bracket installed. Next:"
Write-Host "  Activate the venv:  " -NoNewline; Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Blue
Write-Host "  Launch the UI:      " -NoNewline; Write-Host ".\launch.ps1" -ForegroundColor Blue
Write-Host "  CLI usage:          " -NoNewline; Write-Host "bracket --help" -ForegroundColor Blue
