---
name: bracket-run-session
description: Use when the user wants to run a Bracket orchestration session — picking a trainer, dataset, budget, then kicking off the search. Covers both UI-driven and CLI-driven flows. Use when they say things like "run a Bracket sweep", "fine-tune Z-Image with Bracket", "search for the best LoRA config".
---

# Bracket — running a session

A "session" in Bracket = one full sweep over a configured search space,
producing a Markdown report at the end with the winning config.

## Decide: UI or CLI?

| You want | Use |
|---|---|
| A live dashboard with loss chart, gallery, and a Stop button | UI (`./launch.sh`, then http://127.0.0.1:8000) |
| Headless / scripted / CI | CLI (`bracket --trainer ... --output-dir ...`) |
| Quick test before committing to a full session | CLI with `--budget 2 --max-steps-per-run 50` |

## Required inputs (any flow)

1. **Trainer**: one of `sdxl-lora`, `sdxl-full`, `zimage-lora`, `zimage-full`, `flux2-klein-lora`.
2. **Base model weights** (path to `.safetensors`). For Z-Image and Flux-2 you also need DiT, VAE, and a text encoder.
3. **Dataset TOML** — sd-scripts / musubi-tuner format. Include resolution, batch_size, image_dir, num_repeats.
4. **Output directory** — where the ledger, runs, samples, and `report.md` will land.

## Optional but high-value

- **Sample prompts file** (one prompt per line, sd-scripts format with `--w/--h/--s/--d/--l` flags). Required to enable the VLM judge — without it, runs are scored on training loss only.
- **LMStudio running locally** with a vision model loaded (Qwen3-VL-8B is the default expectation). Confirm via `curl http://localhost:1234/v1/models`.

## CLI invocation (representative)

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

## Choosing a budget

| Budget | Curated runs | Search runs | Wall time (5090, 300 steps each) |
|--------|:-:|:-:|:-:|
| 4 | 4 | 0 | ~30 min |
| 8 | up to N curated for the trainer | rest | ~1 h |
| 16 | curated | TPE-driven | ~2 h |
| 32+ | curated | TPE-driven | overnight |

Recommend **8 with `--seeds-per-config 2`** as the sweet-spot for a real
verdict (gets confidence intervals on the headline result).

## What "the orchestrator does" in order

1. **Setup**: pre-cache latents + text-encoder outputs (one-time per session, cached per dataset).
2. **Baseline**: the trainer's hand-tuned default config.
3. **Curated warm-start**: known-good configs published by each trainer adapter.
4. **Search**: Optuna TPE (or random) explores the rest of the budget.
5. **Finals (optional)**: top-K candidates re-run at higher steps.
6. **Report**: `report.md` written to the session output dir.

## Stopping a session

- UI: click the "Stop" button (visible from any tab when a session is running).
- CLI: `Ctrl+C` — Bracket cleans up the in-flight subprocess and writes a partial report.
- Resuming: re-running with the same `--output-dir` resumes from where the ledger left off (the framework's resume guard verifies the trainer + search-space match).

## Reading the result

Open `<output-dir>/report.md`. Headline shows winner, Δ vs baseline,
confidence (if multi-seed). The ledger at `<output-dir>/ledger.jsonl`
has one line per run with full config + score + judgement breakdown.

## Common failure modes

- **All runs disqualified with `empty_tfevents`**: the trainer subprocess is dying immediately. Check `<output-dir>/runs/<run_id>/logs/stdout.log` for the Python traceback. Most common: an `optimizer_type` choice the trainer doesn't support, or a path it can't find.
- **All runs scored on `loss_only`**: the VLM judge wasn't called. Either `judge_method` wasn't `lmstudio`, or no `sample_prompts` file was provided, or the trainer's sample image filenames don't match Bracket's parser. Run `python scripts/rejudge_run.py <run_dir> <prompts.txt>` to debug.
- **Stuck at 100% on dashboard but no report**: report generation needs at least one scored run. Check ledger.jsonl for any successful row.
