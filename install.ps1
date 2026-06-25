# Bracket installer (Windows PowerShell).
#
# What it does:
#   1. Verifies Python 3.10+ is available.
#   2. Creates a .venv next to the project.
#   3. Detects GPU (nvidia-smi) and selects a matching PyTorch wheel.
#   4. Installs Bracket + its dependencies.
#   5. Builds the React frontend (frontend\dist) if Node.js/npm is present.
#   6. Syncs the trainers Bracket drives (sd-scripts, musubi-tuner, ltx2) as
#      pinned git submodules under vendor\ and creates a shared trainer
#      venv at vendor\venv. The native LTX-2 trainer is uv-managed, so it
#      gets its own .venv via `uv sync` (skipped if uv is absent). Bumping a
#      trainer is a maintainer commit; see docs\UPDATING_TRAINERS.md.
#   7. Writes a .env with sensible defaults so the UI lands ready to run.
#
# Re-run is safe -- every step is idempotent.

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot
$VendorRoot = Join-Path $RepoRoot "vendor"
$VenvDir = Join-Path $RepoRoot ".venv"
$TrainerVenv = Join-Path $VendorRoot "venv"
# Legacy compatibility: if BRACKET_TRAINERS_ROOT was set by an older install,
# we still write it through to .env so existing setups keep working.
$LegacyTrainersRoot = if ($env:BRACKET_TRAINERS_ROOT) { $env:BRACKET_TRAINERS_ROOT } else { $null }

function Write-Step($msg)  { Write-Host "==> $msg" -ForegroundColor Blue }
function Write-Ok($msg)    { Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[!]  $msg" -ForegroundColor Yellow }
function Write-FailExit($msg) {
    Write-Host "[X] $msg" -ForegroundColor Red
    exit 1
}

# Refresh a single KEY=value line in an existing .env file. No-op if
# the current value already matches $Want; otherwise replaces in place
# (or appends if the key is missing). Used to migrate stale trainer
# paths from older install layouts (legacy %LOCALAPPDATA%\bracket\trainers)
# to the in-repo vendor\ submodules without touching unrelated keys
# such as user model paths.
function Refresh-EnvPath {
    param([string]$Key, [string]$Want, [string]$File)
    $lines = Get-Content -LiteralPath $File
    $cur = $null
    foreach ($line in $lines) {
        if ($line -match "^$([regex]::Escape($Key))=(.*)$") { $cur = $Matches[1] }
    }
    if ($cur -eq $Want) { return }
    $replaced = $false
    $newLines = @()
    foreach ($line in $lines) {
        if ($line -match "^$([regex]::Escape($Key))=") {
            $newLines += "$Key=$Want"
            $replaced = $true
        } else {
            $newLines += $line
        }
    }
    if (-not $replaced) { $newLines += "$Key=$Want" }
    Set-Content -LiteralPath $File -Value $newLines -Encoding utf8
    if ($cur) {
        Write-Ok "Refreshed $Key in .env (was: $cur)"
    } else {
        Write-Ok "Added $Key=$Want to .env"
    }
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
        # Driver -> CUDA wheel mapping. cu128 is required for Blackwell
        # GPUs (RTX 5080 / 5090, sm_120) since cu124 wheels stop at sm_90.
        # Detect "RTX 50" in the GPU name as the safest Blackwell signal —
        # driver 570+ is a necessary but not sufficient condition.
        $isBlackwell = $gpuName -match 'RTX\s*50\d{2}'
        if ($driver) {
            $driverMajor = [int]($driver.Split('.')[0])
            if ($isBlackwell -or $driverMajor -ge 570) { $torchIndex = "https://download.pytorch.org/whl/cu128"; $cudaTag = "cu128" }
            elseif ($driverMajor -ge 555) { $torchIndex = "https://download.pytorch.org/whl/cu124"; $cudaTag = "cu124" }
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

# --- 5. Frontend build ------------------------------------------------
Write-Step "Building React frontend"
$frontendDir = Join-Path $RepoRoot "frontend"
if (-not (Test-Path $frontendDir)) {
    Write-Warn "frontend\ directory missing -- skipping UI build"
} else {
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) {
        Write-Warn "npm not found. Install Node.js 20+ from https://nodejs.org/ then re-run this installer to build the React UI. (API endpoints still work without it.)"
    } else {
        $nodeVersion = & node --version 2>$null
        Write-Ok "Node $nodeVersion detected"
        Push-Location $frontendDir
        try {
            $lockfile = Join-Path $frontendDir "package-lock.json"
            if (Test-Path $lockfile) {
                & npm ci --no-audit --no-fund --loglevel=error
            } else {
                & npm install --no-audit --no-fund --loglevel=error
            }
            if ($LASTEXITCODE -ne 0) { Write-FailExit "npm install failed" }
            & npm run build
            if ($LASTEXITCODE -ne 0) { Write-FailExit "npm run build failed" }
            Write-Ok "Frontend built (frontend\dist)"
        } finally {
            Pop-Location
        }
    }
}

# --- 6. Trainers (git submodules) + trainer venv ----------------------
Write-Step "Syncing trainer submodules under vendor\"
$musubiDir = Join-Path $VendorRoot "musubi-tuner"
$sdScriptsDir = Join-Path $VendorRoot "sd-scripts"
$ltx2Dir = Join-Path $VendorRoot "ltx2"
$ltx2TrainerDir = Join-Path $ltx2Dir "packages\ltx-trainer"

# `git submodule update --init --recursive` is idempotent and picks up every
# submodule declared in .gitmodules (musubi-tuner, sd-scripts, ltx2). Pinning
# lives in .gitmodules at the repo root -- bumping a trainer version is a
# maintainer commit, not something this installer does. The ltx2 gitlink is
# created once by a maintainer (see docs\UPDATING_TRAINERS.md); until then
# --init simply no-ops for it.
& git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { Write-FailExit "git submodule update failed" }
if (-not (Test-Path $musubiDir)) { Write-FailExit "vendor\musubi-tuner missing after submodule update" }
if (-not (Test-Path $sdScriptsDir)) { Write-FailExit "vendor\sd-scripts missing after submodule update" }
Write-Ok "Trainer submodules synced ($musubiDir, $sdScriptsDir)"

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
# Install musubi-tuner itself so `python -m musubi_tuner.*` works.
# --ignore-requires-python: musubi's pyproject pins <3.13 conservatively
# but works fine on 3.13. Without this, install silently no-ops on 3.13.
# No --quiet here: any pip failure here breaks every training run, so we
# want it visible.
& $trainerPython -m pip install --ignore-requires-python -e $musubiDir
if ($LASTEXITCODE -ne 0) { Write-FailExit "pip install -e vendor\musubi-tuner failed" }
# Verify the install actually took.
& $trainerPython -c "import musubi_tuner" 2>$null
if ($LASTEXITCODE -ne 0) { Write-FailExit "musubi_tuner not importable after install -- check the pip output above" }
Write-Ok "Trainer deps installed (musubi_tuner importable)"

# --- 6b. Native LTX-2 trainer (uv-managed) ----------------------------
# Lightricks' ltx-trainer is a uv project -- it manages its own .venv and
# dependencies via `uv sync`, separate from the shared vendor\venv above.
# It is optional: if the gitlink hasn't been created yet (see
# docs\UPDATING_TRAINERS.md) or uv isn't installed, we skip it with a
# clear message rather than failing the whole install.
Write-Step "Setting up native LTX-2 trainer (uv-managed)"
if (-not (Test-Path $ltx2TrainerDir)) {
    Write-Warn "vendor\ltx2 not present (submodule gitlink not created yet) -- skipping LTX-2 trainer setup. See docs\UPDATING_TRAINERS.md."
} else {
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        Write-Warn "uv not found -- skipping LTX-2 trainer setup. Install uv (https://docs.astral.sh/uv/) then re-run this installer to enable the native LTX-2 trainer. (Other trainers still work without it.)"
    } else {
        Push-Location $ltx2TrainerDir
        try {
            & uv sync
            if ($LASTEXITCODE -ne 0) { Write-FailExit "uv sync failed in $ltx2TrainerDir" }
            Write-Ok "LTX-2 trainer environment ready (uv sync in $ltx2TrainerDir)"
        } finally {
            Pop-Location
        }
    }
}

# --- 7. .env defaults --------------------------------------------------
Write-Step "Writing .env defaults"
$envFile = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envFile)) {
    $legacyLine = if ($LegacyTrainersRoot) { "BRACKET_TRAINERS_ROOT=$LegacyTrainersRoot`r`n" } else { "" }

    # Auto-detect LMStudio so the VLM judge can unload its model between
    # training runs (otherwise VRAM stays occupied and Z-Image / Flux full
    # FT runs grind to a halt). The lms CLI ships with the LMStudio
    # desktop install; we just record its path. No user action required.
    $lmsCandidate = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe"
    $lmsLine = ""
    if (Test-Path $lmsCandidate) {
        $lmsLine = "BRACKET_LMS_BIN=$lmsCandidate`r`n"
        Write-Ok "Detected LMStudio at $lmsCandidate"
    } else {
        Write-Warn "LMStudio not detected at $lmsCandidate. Install LMStudio if you plan to use the VLM judge (https://lmstudio.ai). Bracket will still install fine."
    }

    $envBody = @"
# Bracket -- environment defaults. Generated by install.ps1.
# Override anything below by editing this file or setting in your shell.

# Trainers are git submodules under vendor/ (pinned in .gitmodules).
# Bumping versions = maintainer commit; users just `git pull` + re-run install.
$legacyLine
BRACKET_VENV_PYTHON=$trainerPython
BRACKET_MUSUBI_DIR=$musubiDir
BRACKET_SD_SCRIPTS_DIR=$sdScriptsDir
$lmsLine

# Add when you have weights downloaded. Each preset only needs the paths
# relevant to its model family -- leave the rest commented.

# --- SDXL --------------------------------------------------------------
# BRACKET_SDXL_PRETRAINED=C:\path\to\sdxl-base-1.0

# --- Z-Image -----------------------------------------------------------
# BRACKET_VAE_PATH=C:\path\to\ae.safetensors
# BRACKET_QWEN3_TE_PATH=C:\path\to\qwen_3_4b.safetensors

# --- Flux-2-Klein ------------------------------------------------------
# BRACKET_FLUX2_DIT_PATH=C:\path\to\flux-2-klein-base-9b-fp8.safetensors
# BRACKET_MISTRAL3_TE_PATH=C:\path\to\mistral_3_small_flux2_fp8.safetensors

# --- Flux.1 + Flux.1-Kontext ------------------------------------------
# BRACKET_FLUX1_DIT_PATH=C:\path\to\flux1-dev.safetensors
# BRACKET_FLUX1_AE_PATH=C:\path\to\ae.safetensors
# BRACKET_FLUX1_KONTEXT_DIT_PATH=C:\path\to\flux1-kontext-dev.safetensors
# BRACKET_T5XXL_PATH=C:\path\to\t5xxl_fp16.safetensors
# BRACKET_CLIP_L_PATH=C:\path\to\clip_l.safetensors

# --- Qwen-Image (TE = Qwen2.5-VL-7B, NOT Qwen3) -----------------------
# BRACKET_QWEN_IMAGE_DIT_PATH=C:\path\to\qwen-image-20b-fp8.safetensors
# BRACKET_QWEN_IMAGE_VAE_PATH=C:\path\to\qwen-image-vae.safetensors
# BRACKET_QWEN_IMAGE_TE_PATH=C:\path\to\qwen2_5_vl_7b.safetensors
# BRACKET_QWEN_IMAGE_EDIT_DIT_PATH=C:\path\to\qwen-image-edit.safetensors

# --- SD3.5 (bundle file contains MMDiT + T5 + CLIPs) ------------------
# BRACKET_SD35_PRETRAINED=C:\path\to\sd3.5_large.safetensors

# --- HunyuanVideo + FramePack (dual TE: LLaMA3 + CLIP-L) --------------
# BRACKET_HUNYUAN_VIDEO_DIT_PATH=C:\path\to\hunyuan-video-13b.safetensors
# BRACKET_HUNYUAN_VIDEO_VAE_PATH=C:\path\to\hunyuan-video-vae.safetensors
# BRACKET_LLAMA3_PATH=C:\path\to\llama3_8b.safetensors
# BRACKET_FRAMEPACK_DIT_PATH=C:\path\to\framepack.safetensors

# --- Wan 2.1 / 2.2 (single TE: UMT5-XXL) ------------------------------
# BRACKET_WAN_DIT_PATH=C:\path\to\wan2.2-14b.safetensors
# BRACKET_WAN_VAE_PATH=C:\path\to\wan-vae.safetensors
# BRACKET_UMT5_PATH=C:\path\to\umt5-xxl.safetensors

# --- LTX-2 (native Lightricks ltx-trainer; Gemma TE, single .safetensors) --
# BRACKET_LTX2_MODEL_PATH=C:\path\to\ltx-2.safetensors
# BRACKET_LTX2_TEXT_ENCODER_PATH=C:\path\to\gemma-text-encoder
# BRACKET_LTX2_TRAINER_DIR=$ltx2TrainerDir

# The presets below need an updated musubi-tuner checkout.

# --- FLUX.2-dev (full FLUX.2; TE = Mistral-3, reuses BRACKET_MISTRAL3_TE_PATH) --
# BRACKET_FLUX2_DEV_DIT_PATH=C:\path\to\flux2-dev.safetensors
# BRACKET_FLUX2_DEV_VAE_PATH=C:\path\to\ae.safetensors

# --- HiDream-O1 (single-file checkpoint; tokenizer/TE auto-load from HF) --
# BRACKET_HIDREAM_DIT_PATH=C:\path\to\hidream-o1.safetensors

# --- HunyuanVideo 1.5 (TE = Qwen2.5-VL + ByT5 glyph encoder) -----------
# BRACKET_HUNYUAN_VIDEO_15_DIT_PATH=C:\path\to\hunyuan-video-1.5-dit.safetensors
# BRACKET_HUNYUAN_VIDEO_15_VAE_PATH=C:\path\to\hunyuan-video-1.5-vae.safetensors
# BRACKET_HUNYUAN_VIDEO_15_QWEN_TE_PATH=C:\path\to\qwen2_5_vl.safetensors
# BRACKET_HUNYUAN_VIDEO_15_BYT5_PATH=C:\path\to\byt5.safetensors

# --- Kandinsky 5 (dual TE: Qwen2.5-VL + CLIP-L) -----------------------
# BRACKET_KANDINSKY5_DIT_PATH=C:\path\to\kandinsky5-dit.safetensors
# BRACKET_KANDINSKY5_VAE_PATH=C:\path\to\kandinsky5-vae.safetensors
# BRACKET_KANDINSKY5_QWEN_TE_PATH=C:\path\to\qwen2_5_vl.safetensors
# BRACKET_KANDINSKY5_CLIP_TE_PATH=C:\path\to\clip_l.safetensors
"@
    Set-Content -Path $envFile -Value $envBody -Encoding utf8
    Write-Ok "Wrote .env"
} else {
    Write-Ok ".env already exists -- refreshing stale trainer paths"
    # Only the three trainer-pointing keys are touched. Model-weight
    # paths (BRACKET_SDXL_PRETRAINED, BRACKET_FLUX1_DIT_PATH, etc.) and
    # any comments are preserved as-is.
    if (Test-Path $TrainerVenv)  { Refresh-EnvPath BRACKET_VENV_PYTHON    $trainerPython $envFile }
    if (Test-Path $musubiDir)    { Refresh-EnvPath BRACKET_MUSUBI_DIR     $musubiDir     $envFile }
    if (Test-Path $sdScriptsDir) { Refresh-EnvPath BRACKET_SD_SCRIPTS_DIR $sdScriptsDir  $envFile }
}

Write-Host ""
Write-Ok "Bracket installed. Next:"
Write-Host "  Activate the venv:  " -NoNewline; Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Blue
Write-Host "  Launch the UI:      " -NoNewline; Write-Host ".\launch.ps1" -ForegroundColor Blue
Write-Host "  CLI usage:          " -NoNewline; Write-Host "bracket --help" -ForegroundColor Blue
