---
name: bracket-quickstart
description: Use when the user asks to install, launch, or first-run Bracket; or asks "how do I get started with Bracket". Walks through install.sh / install.ps1, launches the unified server on port 8000, and verifies the React UI loads.
---

# Bracket — quickstart

When the user wants to install or launch Bracket, follow these steps in order.

## 1. Detect platform

- Linux / macOS / WSL2 → use `install.sh` and `launch.sh`
- Windows native → use `install.ps1` and `launch.ps1` (or the `.bat` wrappers)

## 2. Install

Run from the repo root:

```bash
./install.sh                  # Linux / macOS / WSL2
.\install.ps1                 # Windows PowerShell
install.bat                   # Windows cmd.exe
```

The installer:
- Verifies Python 3.10+
- Detects GPU (`nvidia-smi`) and picks the matching PyTorch wheel
- Creates `.venv/` and installs Bracket editable + dev deps
- Clones `musubi-tuner` and `sd-scripts` into `~/.cache/bracket/trainers/`
  (Windows: `%LOCALAPPDATA%\bracket\trainers\`)
- Creates a shared trainer venv with PyTorch
- Writes `.env` with sensible defaults

If install fails: most likely cause is `git` not on PATH or the repo root
not being writable. Re-running is safe.

## 3. Launch

```bash
./launch.sh                   # Linux / macOS / WSL2
.\launch.ps1                  # Windows PowerShell
launch.bat                    # Windows cmd.exe
```

This starts the FastAPI server (which also serves the built React frontend)
on `http://127.0.0.1:8000`. The browser doesn't auto-open — point at that URL.

## 4. First-run sanity check

Once the UI is up:
1. **Setup tab** should show 3 model families (SDXL, Z-Image, Flux-2-Klein).
2. Pick `Z-Image` → `LoRA`. Required fields appear (`*` marked).
3. Most paths will be empty unless the user has set `BRACKET_*_PATH` env
   vars or has weights downloaded.

If the UI doesn't load: check the server logs (printed to stdout) for
import errors. Common: `pandas` or `pydantic` missing — `pip install
-e .` from the venv usually fixes.

## 5. What to recommend if user has no weights yet

Bracket needs a base model checkpoint to fine-tune against. Point them at:
- SDXL base from Hugging Face (`stabilityai/stable-diffusion-xl-base-1.0`)
- Z-Image base from Tongyi (or Z-Image Turbo for inference-only experimentation; Bracket trains on **base** not Turbo — see CLAUDE.md)
- Flux-2-Klein 9B fp8 from the Flux-2 release

Set the resolved paths via `BRACKET_VAE_PATH`, `BRACKET_QWEN3_TE_PATH`,
etc. in `.env` so they don't have to retype each session.
