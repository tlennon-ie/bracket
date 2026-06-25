---
name: bracket-run-session
description: Use when the user wants to run a Bracket orchestration session — picking a trainer, dataset, budget, then kicking off the search. Covers both UI-driven and CLI-driven flows, including the interactive path where you collect missing inputs (paths, prompts) from the user one question at a time. Use when they say things like "run a Bracket sweep", "fine-tune Z-Image with Bracket", "search for the best LoRA config".
---

# Bracket — running a session (interactive)

A "session" in Bracket = one full sweep over a configured search space,
producing a Markdown report at the end with the winning config.

## Decide: drive the UI, or use the CLI?

| You want | Use |
|---|---|
| A live dashboard with loss chart, gallery, and a Stop button | UI (`./launch.sh`, then http://127.0.0.1:8000) |
| Headless / scripted / CI | CLI (`bracket --trainer ... --output-dir ...`) |
| Quick sanity test before a real session | CLI with `--budget 2 --max-steps-per-run 50` |
| Drive end-to-end from chat with you collecting paths from the user | Use the `bracket-session-agent` subagent (in `.claude/agents/`) |

## Interactive flow — collect inputs one at a time

When the user says "run a Bracket session", DON'T dump the full
required-inputs list in one wall of text. Walk them through it:

1. **Confirm a trainer family** first. Ask which model (SDXL, Z-Image,
   Flux.1, Flux-2-Klein, Qwen-Image, SD3.5, HunyuanVideo, Wan, LTX-2,
   FramePack), then LoRA vs Full FT. Many families don't have a
   Full-FT preset (Flux-2-Klein, Flux.1-Kontext, Qwen-Image-Edit, Wan,
   LTX-2, FramePack) — if they pick those, the answer is LoRA
   only.

2. **Confirm weights are downloaded.** Ask for the path(s) the trainer
   needs:

   | Family | Ask for |
   |---|---|
   | SDXL | base model directory or single .safetensors |
   | Z-Image | DiT, VAE, Qwen3 TE |
   | Flux-2-Klein | DiT, VAE, Mistral-3-Small TE |
   | Flux.1 / Flux.1-Kontext | DiT (or Kontext DiT), AE, T5-XXL, CLIP-L |
   | Qwen-Image / Qwen-Image-Edit | DiT, VAE, Qwen2.5-VL-7B TE (NOT Qwen3) |
   | SD3.5 | base file (MMDiT + T5 + CLIPs bundle) |
   | HunyuanVideo / FramePack | DiT, VAE, LLaMA3, CLIP-L |
   | Wan 2.1 / 2.2 | DiT, VAE, UMT5-XXL |
   | LTX-2 | model path, Gemma TE (native ltx-trainer) |

   Check `.env` first — many users have these set as `BRACKET_*_PATH`
   env vars. If found, just confirm; don't re-ask.

3. **Confirm a dataset.toml** exists. If not → hand off to
   `bracket-dataset-toml` skill which builds one interactively from a
   directory of images + captions.

4. **Sample prompts file.** Ask if the user wants the VLM judge.
   - Yes → ask for a path to a prompts file (one prompt per line in
     sd-scripts format with `--w/--h/--s/--d/--l` flags). If they
     don't have one, offer to write a 5-prompt example file matching
     their dataset.
   - No → fall back to loss-only scoring (mention this so they know
     it's less informative).

5. **Output directory.** Default to
   `./runs/<dataset-name>-<timestamp>`. Confirm.

6. **Budget.** Recommend 8 with `--seeds-per-config 2` as the
   sweet-spot for a real verdict. Cheaper options: 4 / single-seed for
   "just see if this works".

7. **VRAM tier.** Auto-detect via `nvidia-smi` if possible (on
   Windows: `nvidia-smi --query-gpu=memory.total --format=csv`). If
   you can't shell out, ask the user: 8 / 12 / 16 / 24 GB+. Bracket
   uses this to filter the search space (smaller VRAM = no big batch,
   no high blocks_to_swap).

Only AFTER all six are answered, recap the full configuration in a
fenced block and ask for one final "OK to start?".

## Required inputs (any flow)

1. **Trainer**: `sdxl-lora`, `sdxl-full`, `zimage-lora`, `zimage-full`,
   `flux2-klein-lora`, `flux1-lora`, `flux1-full`,
   `flux1-kontext-lora`, `qwen-image-lora`, `qwen-image-full`,
   `qwen-image-edit-lora`, `sd35-lora`, `sd35-full`,
   `hunyuan-video-lora`, `hunyuan-video-full`, `wan22-lora`,
   `wan21-lora`, `ltx2-t2v-lora`, `ltx2-i2v-lora`, `framepack-lora`.
2. **Base model weights** — depends on family (table above).
3. **Dataset TOML** — sd-scripts / musubi-tuner format. See
   `bracket-dataset-toml` skill for how to write one.
4. **Output directory** — where the ledger, runs, samples, and
   `report.md` will land.

## Optional but high-value

- **Sample prompts file** (one prompt per line, sd-scripts format with
  `--w/--h/--s/--d/--l` flags). Required to enable the VLM judge —
  without it, runs are scored on training loss only.
- **LMStudio running locally** with a vision model loaded (Qwen3-VL
  variants, Qwen2.5-VL, MiniCPM-V all work). Confirm via
  `curl http://localhost:1234/v1/models`. Make sure
  `BRACKET_LMS_BIN` is set in `.env` so VRAM is released between
  runs (the installer auto-detects it; see `bracket-debug-run` for
  the failure mode if it isn't).

## CLI invocation (representative — Z-Image full FT)

```bash
bracket \
    --trainer zimage-full \
    --dataset-toml ./configs/portraits.toml \
    --sample-prompts ./configs/prompts.txt \
    --output-dir ./runs/portraits-001 \
    --dit "$BRACKET_FLUX2_DIT_PATH" \
    --vae "$BRACKET_VAE_PATH" \
    --text-encoder "$BRACKET_QWEN3_TE_PATH" \
    --budget 8 \
    --seeds-per-config 2 \
    --max-steps-per-run 300 \
    --judge lmstudio \
    --finals-top-k 3 --finals-max-steps 1500
```

For other trainers, swap `--trainer` and the model-weight flags
(`--dit`, `--vae`, `--text-encoder`, `--t5xxl`, `--clip-l`, etc. —
`bracket --help` lists the trainer-specific flag set per preset).

## Choosing a budget

| Budget | Curated runs | Search runs | Wall time (5090, 300 steps each) |
|--------|:-:|:-:|:-:|
| 4 | up to 4 curated | 0 | ~30 min |
| 8 | curated for the trainer | rest | ~1 h |
| 16 | curated | TPE-driven | ~2 h |
| 32+ | curated | TPE-driven | overnight |

Recommend **8 with `--seeds-per-config 2`** as the sweet-spot for a
real verdict (gets confidence intervals on the headline result).

## What "the orchestrator does" in order

1. **Setup**: pre-cache latents + text-encoder outputs (one-time per
   session, cached per dataset). Skipped for sd-scripts trainers
   (they cache on the fly).
2. **Baseline**: the trainer's hand-tuned default config.
3. **Curated warm-start**: known-good configs published by each
   trainer adapter.
4. **Search**: Optuna TPE (or random) explores the rest of the budget.
5. **Finals (optional)**: top-K candidates re-run at higher steps.
6. **Report**: `report.md` written to the session output dir.

## Stopping a session

- UI: click the "Stop" button (visible from any tab when a session is
  running).
- CLI: `Ctrl+C` — Bracket cleans up the in-flight subprocess and
  writes a partial report.
- Resuming: re-running with the same `--output-dir` resumes from where
  the ledger left off (the framework's resume guard verifies the
  trainer + search-space match).

## Reading the result

Open `<output-dir>/report.md`. Headline shows winner, Δ vs baseline,
confidence (if multi-seed). The ledger at `<output-dir>/ledger.jsonl`
has one line per run with full config + score + judgement breakdown.

## When something goes wrong

Hand off to the `bracket-debug-run` skill. The most common failure —
"first run finishes, all subsequent runs hit the wall-clock with
n_steps: 3" — is the LM Studio VRAM-leak; that skill walks the user
through the fix.
