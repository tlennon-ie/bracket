#!/usr/bin/env bash
# Bracket installer (Linux / macOS / WSL2).
#
# What it does:
#   1. Verifies Python 3.10+ is available.
#   2. Creates a .venv next to the project.
#   3. Detects GPU (nvidia-smi) and selects a matching PyTorch wheel.
#   4. Installs Bracket + its dependencies.
#   5. Clones the trainers Bracket drives (sd-scripts, musubi-tuner) into
#      ~/.cache/bracket/trainers/ and creates a shared trainer venv.
#   6. Writes a .env with sensible defaults so the UI lands ready to run.
#
# Re-run is safe — every step is idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINERS_ROOT="${BRACKET_TRAINERS_ROOT:-$HOME/.cache/bracket/trainers}"
VENV_DIR="$REPO_ROOT/.venv"
TRAINER_VENV="$TRAINERS_ROOT/venv"

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
        # Crude CUDA-version inference from driver — see PyTorch wheel matrix
        if [ -n "${DRIVER:-}" ]; then
            DRIVER_MAJOR=${DRIVER%%.*}
            if [ "$DRIVER_MAJOR" -ge 555 ]; then
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
        GPU_DESC="$GPU_NAME (driver $DRIVER → ${CUDA_TAG:-cpu})"
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

# ─── 5. Trainers + trainer venv ────────────────────────────────────
step "Setting up trainers in $TRAINERS_ROOT"
mkdir -p "$TRAINERS_ROOT"
if [ ! -d "$TRAINERS_ROOT/musubi-tuner" ]; then
    git clone --depth=1 https://github.com/kohya-ss/musubi-tuner "$TRAINERS_ROOT/musubi-tuner"
    ok "Cloned musubi-tuner"
else
    ok "musubi-tuner already present"
fi
if [ ! -d "$TRAINERS_ROOT/musubi-tuner/sd-scripts" ]; then
    git clone --depth=1 https://github.com/kohya-ss/sd-scripts "$TRAINERS_ROOT/musubi-tuner/sd-scripts"
    ok "Cloned sd-scripts"
else
    ok "sd-scripts already present"
fi

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
if [ -f "$TRAINERS_ROOT/musubi-tuner/requirements.txt" ]; then
    python -m pip install --quiet -r "$TRAINERS_ROOT/musubi-tuner/requirements.txt"
fi
if [ -f "$TRAINERS_ROOT/musubi-tuner/sd-scripts/requirements.txt" ]; then
    python -m pip install --quiet -r "$TRAINERS_ROOT/musubi-tuner/sd-scripts/requirements.txt"
fi
ok "Trainer deps installed"

deactivate

# ─── 6. .env defaults ──────────────────────────────────────────────
step "Writing .env defaults"
ENV_FILE="$REPO_ROOT/.env"
if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<EOF
# Bracket — environment defaults. Generated by install.sh.
# Override anything below by editing this file or exporting in your shell.

BRACKET_TRAINERS_ROOT=$TRAINERS_ROOT
BRACKET_VENV_PYTHON=$TRAINER_VENV/bin/python
BRACKET_MUSUBI_DIR=$TRAINERS_ROOT/musubi-tuner
BRACKET_SD_SCRIPTS_DIR=$TRAINERS_ROOT/musubi-tuner/sd-scripts

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

# ─── LTX-Video ──────────────────────────────────────────────────────
# BRACKET_LTX_VIDEO_DIT_PATH=/abs/path/to/ltx-video-dit.safetensors
# BRACKET_LTX_VIDEO_VAE_PATH=/abs/path/to/ltx-video-vae.safetensors
EOF
    ok "Wrote .env"
else
    ok ".env already exists — leaving alone"
fi

echo
ok "Bracket installed. Next:"
printf "  Activate the venv:    %ssource .venv/bin/activate%s\n" "$c_blue" "$c_reset"
printf "  Launch the UI:        %s./launch.sh%s\n" "$c_blue" "$c_reset"
printf "  CLI usage:            %sbracket --help%s\n" "$c_blue" "$c_reset"
