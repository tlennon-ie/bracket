---
name: bracket-quickstart
description: Use when the user asks to install, launch, or first-run Bracket; or asks "how do I get started with Bracket". Walks through install.sh / install.ps1, launches the unified server on port 8000, and verifies the React UI loads.
---

# Bracket — quickstart

When the user wants to install or launch Bracket, follow these steps in
order.

## 1. Detect platform

- Linux / macOS / WSL2 → use `install.sh` and `launch.sh`
- Windows native → use `install.ps1` and `launch.ps1` (or the `.bat`
  wrappers `install.bat` / `launch.bat`)

## 2. Install

Run from the repo root:

```bash
./install.sh                  # Linux / macOS / WSL2
.\install.ps1                 # Windows PowerShell
install.bat                   # Windows cmd.exe
```

The installer:
- Verifies Python 3.10+.
- Detects GPU via `nvidia-smi` and picks the matching PyTorch wheel
  (`cu124` / `cu121` / `cu118` / CPU).
- Creates `.venv/` and installs Bracket editable + dev deps.
- Detects Node.js and runs `npm ci && npm run build` to build the React
  bundle into `frontend/dist/` (warn-and-skip if Node missing).
- Syncs `vendor/musubi-tuner` and `vendor/sd-scripts` git submodules
  pinned to specific commits in `.gitmodules`.
- Creates a shared trainer venv at `vendor/venv/` with PyTorch +
  trainer requirements + editable install of musubi-tuner.
- Detects LMStudio at `~/.lmstudio/bin/lms[.exe]` and writes
  `BRACKET_LMS_BIN` to `.env` (needed so the VLM judge can release
  VRAM between runs — see `bracket-debug-run`).
- Writes `.env` with sensible defaults.

If install fails: most likely causes are `git` / Node / Python missing
from PATH, or the repo root not being writable. Re-running is safe —
every step is idempotent.

## 3. Launch

```bash
./launch.sh                   # Linux / macOS / WSL2
.\launch.ps1                  # Windows PowerShell
launch.bat                    # Windows cmd.exe
```

This starts FastAPI (with the built React SPA mounted at `/`) on
`http://127.0.0.1:8000`. The browser doesn't auto-open — point at that
URL.

If the URL returns 404 instead of the UI: the React bundle isn't built.
Run `cd frontend && npm run build` then refresh, or re-run the
installer.

## 4. First-run sanity check

Once the UI is up:
1. **Setup tab** should show many model families (SDXL, Z-Image,
   Flux-2-Klein, Flux.1, Flux.1-Kontext, Qwen-Image, Qwen-Image-Edit,
   SD3.5, HunyuanVideo, Wan 2.2, Wan 2.1, LTX-2, FramePack) plus the
   ai-toolkit-backed ones (Chroma, Lumina2, OmniGen2, Flex.1, Flex.2,
   LTX-2.5, MiniMax-H3, MiniMax-H3 Ref2VA, ACE-Step 1.5 / 1.5 XL).
2. Pick any one → `LoRA` or `Full FT`. Required fields appear (`*`
   marked).
3. Trainer-infrastructure paths (musubi-tuner directory, sd-scripts
   directory, trainer venv python) should be **prefilled** from the
   `vendor/` submodules — the user shouldn't have to type them.
4. Model-weight paths (DiT / VAE / TE) will be empty unless the user
   has set the corresponding `BRACKET_*_PATH` env vars in `.env`.

If the UI doesn't load: check the server logs (printed to stdout) for
import errors. Common: `pandas` or `pydantic` missing — `pip install
-e .` from `.venv` usually fixes.

## 5. What to recommend if the user has no weights yet

Bracket needs a base model checkpoint to fine-tune against. Each preset
in the UI lists the env-var name in its help text. Common ones:

| Family | Env vars to populate |
|---|---|
| SDXL | `BRACKET_SDXL_PRETRAINED` |
| Z-Image | `BRACKET_VAE_PATH`, `BRACKET_QWEN3_TE_PATH`, plus DiT path per session |
| Flux-2-Klein | `BRACKET_FLUX2_DIT_PATH`, `BRACKET_VAE_PATH`, `BRACKET_MISTRAL3_TE_PATH` |
| Flux.1 / Flux.1-Kontext | `BRACKET_FLUX1_DIT_PATH`, `BRACKET_FLUX1_AE_PATH`, `BRACKET_T5XXL_PATH`, `BRACKET_CLIP_L_PATH` (+ `BRACKET_FLUX1_KONTEXT_DIT_PATH` for Kontext) |
| Qwen-Image / Qwen-Image-Edit | `BRACKET_QWEN_IMAGE_DIT_PATH`, `BRACKET_QWEN_IMAGE_VAE_PATH`, `BRACKET_QWEN_IMAGE_TE_PATH` (TE is **Qwen2.5-VL-7B**, NOT Qwen3) |
| SD3.5 | `BRACKET_SD35_PRETRAINED` (single file with MMDiT + T5 + CLIPs) |
| HunyuanVideo / FramePack | `BRACKET_HUNYUAN_VIDEO_DIT_PATH`, `BRACKET_HUNYUAN_VIDEO_VAE_PATH`, `BRACKET_LLAMA3_PATH`, `BRACKET_CLIP_L_PATH` |
| Wan 2.1 / 2.2 | `BRACKET_WAN_DIT_PATH`, `BRACKET_WAN_VAE_PATH`, `BRACKET_UMT5_PATH` |
| LTX-2 | `BRACKET_LTX2_MODEL_PATH`, `BRACKET_LTX2_TEXT_ENCODER_PATH` (native ltx-trainer) |

The **ai-toolkit-backed** presets (Chroma, Lumina2, OmniGen2, Flex.1/2,
LTX-2.5, MiniMax-H3 / Ref2VA, ACE-Step 1.5 / XL) take no `BRACKET_*_PATH` env
vars — they use a single "model (HF id or path)" field prefilled with the
model's canonical HF repo, and ai-toolkit downloads missing components into
its own models folder on first load. Two of them need a prerequisite:

- **LTX-2.5** — `Lightricks/LTX-2.5` is a gated repo. Accept the licence on
  Hugging Face, then `huggingface-cli login`, before the first run.
- **MiniMax-H3** — pulls the pre-quantised `Comfy-Org/MiniMax-H3` weights
  (int8-ConvRot DiT + nvfp4 TE) plus ostris' training adapter. Expect a large
  first-run download.

Don't bundle weights — they're 5-25 GB per family. Tell the user to
download the ones they need from Hugging Face / the model's release
page, then either set the env vars in `.env` or fill the path field in
the Setup tab once per session.

## 6. Want to run a session next?

Hand off to the `bracket-run-session` skill. It walks through dataset
selection, judge configuration, budget, and stage-2 finals.

## 7. Want to build a dataset.toml?

Hand off to the `bracket-dataset-toml` skill. It generates a valid
musubi/sd-scripts dataset config from a directory of images +
captions, asking the user only for the paths it can't infer.
