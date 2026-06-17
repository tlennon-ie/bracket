#!/usr/bin/env bash
# Bracket installer (Linux / macOS / WSL2).
#
# What it does:
#   1. Verifies Python 3.10+ is available.
#   2. Creates a .venv next to the project.
#   3. Detects GPU (nvidia-smi) and selects a matching PyTorch wheel.
#   4. Installs Bracket + its dependencies.
#   5. Builds the React frontend (frontend/dist) if Node.js/npm is present.
#   6. Syncs the trainers Bracket drives (sd-scripts, musubi-tuner, ltx2) as
#      pinned git submodules under vendor/ and creates a shared trainer
#      venv at vendor/venv. The native LTX-2 trainer is uv-managed, so it
#      gets its own .venv via `uv sync` (skipped if uv is absent). Bumping a
#      trainer is a maintainer commit; see docs/UPDATING_TRAINERS.md.
#   7. Writes a .env with sensible defaults so the UI lands ready to run.
#
# Re-run is safe — every step is idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_ROOT="$REPO_ROOT/vendor"
VENV_DIR="$REPO_ROOT/.venv"
TRAINER_VENV="$VENDOR_ROOT/venv"
# Legacy compatibility: write BRACKET_TRAINERS_ROOT into .env only if the
# user already had it set (older installs targeted ~/.cache/bracket/trainers).
LEGACY_TRAINERS_ROOT="${BRACKET_TRAINERS_ROOT:-}"

c_blue=$'\033[0;34m'
c_green=$'\033[0;32m'
c_yellow=$'\033[1;33m'
c_red=$'\033[0;31m'
c_reset=$'\033[0m'

step() { printf "%s>%s %s\n" "$c_blue" "$c_reset" "$*"; }
ok() { printf "%s✓%s %s\n" "$c_green" "$c_reset" "$*"; }
warn() { printf "%s!%s %s\n" "$c_yellow" "$c_reset" "$*" >&2; }
fail() { printf "%s✗%s %s\n" "$c_red" "$c_reset" "$*" >&2; exit 1; }

# ─── 1. Python version ─────────────────────────────────────────────
step "Checking Python"
if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found. Install Python 3.10+ from https://www.python.org/"
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=${PY_VERSION%.*}
PY_MINOR=${PY_VERSION#*.}
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    fail "Python $PY_VERSION found, but Bracket needs >=3.10."
fi
ok "Python $PY_VERSION"

# ─── 2. Hardware detection ─────────────────────────────────────────
step "Detecting hardware"
TORCH_INDEX=""
GPU_DESC="CPU only"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    if [ -n "${GPU_NAME:-}" ]; then
        # Driver -> CUDA wheel mapping. cu128 is required for Blackwell
        # GPUs (RTX 5080 / 5090, sm_120) since cu124 wheels stop at sm_90.
        # Detect "RTX 50" in the GPU name as the safest Blackwell signal —
        # driver 570+ is a necessary but not sufficient condition.
        IS_BLACKWELL=0
        case "$GPU_NAME" in *"RTX 50"*) IS_BLACKWELL=1 ;; esac
        if [ -n "${DRIVER:-}" ]; then
            DRIVER_MAJOR=${DRIVER%%.*}
            if [ "$IS_BLACKWELL" -eq 1 ] || [ "$DRIVER_MAJOR" -ge 570 ]; then
                TORCH_INDEX="https://download.pytorch.org/whl/cu128"
                CUDA_TAG="cu128"
            elif [ "$DRIVER_MAJOR" -ge 555 ]; then
                TORCH_INDEX="https://download.pytorch.org/whl/cu124"
                CUDA_TAG="cu124"
            elif [ "$DRIVER_MAJOR" -ge 525 ]; then
                TORCH_INDEX="https://download.pytorch.org/whl/cu121"
                CUDA_TAG="cu121"
            else
                TORCH_INDEX="https://download.pytorch.org/whl/cu118"
                CUDA_TAG="cu118"
            fi
        fi
        GPU_DESC="$GPU_NAME (driver $DRIVER -> ${CUDA_TAG:-cpu})"
    fi
fi
ok "GPU: $GPU_DESC"

# ─── 3. Repo venv ──────────────────────────────────────────────────
step "Creating repo venv at .venv/"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --quiet --upgrade pip wheel
ok "Repo venv ready"

# ─── 4. Bracket install ────────────────────────────────────────────
step "Installing Bracket (editable + dev)"
python -m pip install --quiet -e ".[dev]"
ok "Bracket installed"

# ─── 5. Frontend build ─────────────────────────────────────────────
step "Building React frontend"
FRONTEND_DIR="$REPO_ROOT/frontend"
if [ ! -d "$FRONTEND_DIR" ]; then
    warn "frontend/ directory missing — skipping UI build"
elif ! command -v npm >/dev/null 2>&1; then
    warn "npm not found. Install Node.js 20+ from https://nodejs.org/ then re-run this installer to build the React UI. (API endpoints still work without it.)"
else
    NODE_VERSION=$(node --version 2>/dev/null || echo "unknown")
    ok "Node $NODE_VERSION detected"
    (
        cd "$FRONTEND_DIR"
        if [ -f package-lock.json ]; then
            npm ci --no-audit --no-fund --loglevel=error
        else
            npm install --no-audit --no-fund --loglevel=error
        fi
        npm run build
    )
    ok "Frontend built (frontend/dist)"
fi

# ─── 6. Trainers (git submodules) + trainer venv ───────────────────
step "Syncing trainer submodules under vendor/"
MUSUBI_DIR="$VENDOR_ROOT/musubi-tuner"
SD_SCRIPTS_DIR="$VENDOR_ROOT/sd-scripts"
LTX2_DIR="$VENDOR_ROOT/ltx2"
LTX2_TRAINER_DIR="$LTX2_DIR/packages/ltx-trainer"

# `git submodule update --init --recursive` is idempotent and picks up every
# submodule declared in .gitmodules (musubi-tuner, sd-scripts, ltx2). Pinned
# commits live in .gitmodules at the repo root — bumping a trainer is a
# maintainer commit, not something this installer does. The ltx2 gitlink is
# created once by a maintainer (see docs/UPDATING_TRAINERS.md); until then
# --init simply no-ops for it. See docs/UPDATING_TRAINERS.md.
git submodule update --init --recursive
[ -d "$MUSUBI_DIR" ] || fail "vendor/musubi-tuner missing after submodule update"
[ -d "$SD_SCRIPTS_DIR" ] || fail "vendor/sd-scripts missing after submodule update"
ok "Trainer submodules synced ($MUSUBI_DIR, $SD_SCRIPTS_DIR)"

# Trainer venv (shared by all trainers — they have compatible deps).
if [ ! -d "$TRAINER_VENV" ]; then
    python3 -m venv "$TRAINER_VENV"
fi
# shellcheck disable=SC1091
source "$TRAINER_VENV/bin/activate"
python -m pip install --quiet --upgrade pip wheel

# Install PyTorch matching the detected CUDA, falling back to CPU.
step "Installing PyTorch into trainer venv ($GPU_DESC)"
if [ -n "$TORCH_INDEX" ]; then
    python -m pip install --quiet torch torchvision --index-url "$TORCH_INDEX"
else
    python -m pip install --quiet torch torchvision
    warn "No NVIDIA GPU detected — installed CPU-only PyTorch. Training will be very slow."
fi
ok "PyTorch installed"

step "Installing trainer dependencies"
if [ -f "$MUSUBI_DIR/requirements.txt" ]; then
    python -m pip install --quiet -r "$MUSUBI_DIR/requirements.txt"
fi
if [ -f "$SD_SCRIPTS_DIR/requirements.txt" ]; then
    python -m pip install --quiet -r "$SD_SCRIPTS_DIR/requirements.txt"
fi
# Install musubi-tuner itself so `python -m musubi_tuner.*` works.
# --ignore-requires-python: musubi's pyproject pins <3.13 conservatively
# but works fine on 3.13. Without this, install silently no-ops on 3.13.
# No --quiet here: any pip failure here breaks every training run, so we
# want it visible.
python -m pip install --ignore-requires-python -e "$MUSUBI_DIR"
python -c "import musubi_tuner" >/dev/null 2>&1 \
    || fail "musubi_tuner not importable after install -- check pip output above"
ok "Trainer deps installed (musubi_tuner importable)"

deactivate

# ─── 6b. Native LTX-2 trainer (uv-managed) ─────────────────────────
# Lightricks' ltx-trainer is a uv project — it manages its own .venv and
# dependencies via `uv sync`, separate from the shared vendor/venv above.
# It is optional: if the gitlink hasn't been created yet (see
# docs/UPDATING_TRAINERS.md) or uv isn't installed, we skip it with a
# clear message rather than failing the whole install.
step "Setting up native LTX-2 trainer (uv-managed)"
if [ ! -d "$LTX2_TRAINER_DIR" ]; then
    warn "vendor/ltx2 not present (submodule gitlink not created yet) — skipping LTX-2 trainer setup. See docs/UPDATING_TRAINERS.md."
elif ! command -v uv >/dev/null 2>&1; then
    warn "uv not found — skipping LTX-2 trainer setup. Install uv (https://docs.astral.sh/uv/) then re-run this installer to enable the native LTX-2 trainer. (Other trainers still work without it.)"
else
    (
        cd "$LTX2_TRAINER_DIR" || exit 1
        uv sync
    ) || fail "uv sync failed in $LTX2_TRAINER_DIR — see output above"
    ok "LTX-2 trainer environment ready (uv sync in $LTX2_TRAINER_DIR)"
fi

# Refresh a single KEY=value line in an existing .env file. No-op if
# the current value already matches $want; otherwise replaces in place
# (or appends if the key is missing). Used to migrate stale trainer
# paths from older install layouts (e.g. legacy ~/.cache/bracket/trainers
# / %LOCALAPPDATA%\bracket\trainers) to the in-repo vendor/ submodules
# without touching unrelated keys such as user model paths.
refresh_env_path() {
    local key="$1" want="$2" file="$3"
    local cur=""
    cur=$(grep -E "^${key}=" "$file" | tail -1 | sed -E "s|^${key}=||" || true)
    if [ "$cur" = "$want" ]; then return 0; fi
    if grep -qE "^${key}=" "$file"; then
        local tmp="${file}.tmp.$$"
        awk -v k="$key" -v v="$want" '
            $0 ~ "^"k"=" {print k"="v; next}
            {print}
        ' "$file" > "$tmp" && mv "$tmp" "$file"
    else
        printf "%s=%s\n" "$key" "$want" >> "$file"
    fi
    if [ -n "$cur" ]; then
        ok "Refreshed $key in .env (was: $cur)"
    else
        ok "Added $key=$want to .env"
    fi
}

# ─── 7. .env defaults ──────────────────────────────────────────────
step "Writing .env defaults"
ENV_FILE="$REPO_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    legacy_line=""
    if [ -n "$LEGACY_TRAINERS_ROOT" ]; then
        legacy_line="BRACKET_TRAINERS_ROOT=$LEGACY_TRAINERS_ROOT"
    fi

    # Auto-detect LMStudio so the VLM judge can unload its model between
    # training runs (otherwise VRAM stays occupied and Z-Image / Flux full
    # FT runs grind to a halt). The lms CLI ships with the LMStudio
    # desktop install; we just record its path. No user action required.
    lms_line=""
    LMS_CANDIDATE="$HOME/.lmstudio/bin/lms"
    if [ -f "$LMS_CANDIDATE" ]; then
        lms_line="BRACKET_LMS_BIN=$LMS_CANDIDATE"
        ok "Detected LMStudio at $LMS_CANDIDATE"
    else
        warn "LMStudio not detected at $LMS_CANDIDATE. Install LMStudio if you plan to use the VLM judge (https://lmstudio.ai). Bracket will still install fine."
    fi

    cat > "$ENV_FILE" <<EOF
# Bracket — environment defaults. Generated by install.sh.
# Override anything below by editing this file or exporting in your shell.

# Trainers are git submodules under vendor/ (pinned in .gitmodules).
# Bumping versions = maintainer commit; users just \`git pull\` + re-run install.
$legacy_line
BRACKET_VENV_PYTHON=$TRAINER_VENV/bin/python
BRACKET_MUSUBI_DIR=$MUSUBI_DIR
BRACKET_SD_SCRIPTS_DIR=$SD_SCRIPTS_DIR
$lms_line

# Add when you have weights downloaded. Each preset only needs the paths
# relevant to its model family — leave the rest commented.

# ─── SDXL ────────────────────────────────────────────────────────────
# BRACKET_SDXL_PRETRAINED=/abs/path/to/sdxl-base-1.0

# ─── Z-Image (Tongyi-MAI) ────────────────────────────────────────────
# BRACKET_VAE_PATH=/abs/path/to/ae.safetensors
# BRACKET_QWEN3_TE_PATH=/abs/path/to/qwen_3_4b.safetensors

# ─── Flux-2-Klein ────────────────────────────────────────────────────
# BRACKET_FLUX2_DIT_PATH=/abs/path/to/flux-2-klein-base-9b-fp8.safetensors
# BRACKET_MISTRAL3_TE_PATH=/abs/path/to/mistral_3_small_flux2_fp8.safetensors

# ─── Flux.1 + Flux.1-Kontext ─────────────────────────────────────────
# BRACKET_FLUX1_DIT_PATH=/abs/path/to/flux1-dev.safetensors
# BRACKET_FLUX1_AE_PATH=/abs/path/to/ae.safetensors
# BRACKET_FLUX1_KONTEXT_DIT_PATH=/abs/path/to/flux1-kontext-dev.safetensors
# BRACKET_T5XXL_PATH=/abs/path/to/t5xxl_fp16.safetensors
# BRACKET_CLIP_L_PATH=/abs/path/to/clip_l.safetensors

# ─── Qwen-Image (TE = Qwen2.5-VL-7B, NOT Qwen3) ─────────────────────
# BRACKET_QWEN_IMAGE_DIT_PATH=/abs/path/to/qwen-image-20b-fp8.safetensors
# BRACKET_QWEN_IMAGE_VAE_PATH=/abs/path/to/qwen-image-vae.safetensors
# BRACKET_QWEN_IMAGE_TE_PATH=/abs/path/to/qwen2_5_vl_7b.safetensors
# BRACKET_QWEN_IMAGE_EDIT_DIT_PATH=/abs/path/to/qwen-image-edit.safetensors

# ─── SD3.5 (bundle file contains MMDiT + T5 + CLIPs) ────────────────
# BRACKET_SD35_PRETRAINED=/abs/path/to/sd3.5_large.safetensors

# ─── HunyuanVideo + FramePack (dual TE: LLaMA3 + CLIP-L) ────────────
# BRACKET_HUNYUAN_VIDEO_DIT_PATH=/abs/path/to/hunyuan-video-13b.safetensors
# BRACKET_HUNYUAN_VIDEO_VAE_PATH=/abs/path/to/hunyuan-video-vae.safetensors
# BRACKET_LLAMA3_PATH=/abs/path/to/llama3_8b.safetensors
# BRACKET_FRAMEPACK_DIT_PATH=/abs/path/to/framepack.safetensors

# ─── Wan 2.1 / 2.2 (single TE: UMT5-XXL) ────────────────────────────
# BRACKET_WAN_DIT_PATH=/abs/path/to/wan2.2-14b.safetensors
# BRACKET_WAN_VAE_PATH=/abs/path/to/wan-vae.safetensors
# BRACKET_UMT5_PATH=/abs/path/to/umt5-xxl.safetensors

# ─── LTX-Video (musubi-tuner backend) ───────────────────────────────
# BRACKET_LTX_VIDEO_DIT_PATH=/abs/path/to/ltx-video-dit.safetensors
# BRACKET_LTX_VIDEO_VAE_PATH=/abs/path/to/ltx-video-vae.safetensors

# ─── LTX-2 (native Lightricks ltx-trainer; Gemma TE, single .safetensors) ──
# BRACKET_LTX2_MODEL_PATH=/abs/path/to/ltx-2.safetensors
# BRACKET_LTX2_TEXT_ENCODER_PATH=/abs/path/to/gemma-text-encoder
# BRACKET_LTX2_TRAINER_DIR=$LTX2_TRAINER_DIR
EOF
    ok "Wrote .env"
else
    ok ".env already exists — refreshing stale trainer paths"
    # Only the three trainer-pointing keys are touched. Model-weight
    # paths (BRACKET_SDXL_PRETRAINED, BRACKET_FLUX1_DIT_PATH, etc.) and
    # any comments are preserved as-is.
    [ -d "$TRAINER_VENV"  ] && refresh_env_path BRACKET_VENV_PYTHON    "$TRAINER_VENV/bin/python" "$ENV_FILE"
    [ -d "$MUSUBI_DIR"    ] && refresh_env_path BRACKET_MUSUBI_DIR     "$MUSUBI_DIR"              "$ENV_FILE"
    [ -d "$SD_SCRIPTS_DIR" ] && refresh_env_path BRACKET_SD_SCRIPTS_DIR "$SD_SCRIPTS_DIR"         "$ENV_FILE"
fi

echo
ok "Bracket installed. Next:"
printf "  Activate the venv:    %ssource .venv/bin/activate%s\n" "$c_blue" "$c_reset"
printf "  Launch the UI:        %s./launch.sh%s\n" "$c_blue" "$c_reset"
printf "  CLI usage:            %sbracket --help%s\n" "$c_blue" "$c_reset"
