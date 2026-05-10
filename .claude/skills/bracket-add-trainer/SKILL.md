---
name: bracket-add-trainer
description: Use when the user wants to add support for a new diffusion model or LLM training script in Bracket. Covers the Trainer protocol, search-space declaration, curated configs, presets, and tests. Lists every currently-supported preset so you know whether the model is already wired in.
---

# Bracket — adding a new trainer adapter

Bracket drives external trainers via subprocess. To support a new one,
implement the `Trainer` protocol and register a preset. ~150 lines for
SDXL-class models; same shape for everything else.

## Step 0: Is the model already supported?

Check the list before authoring anything new — the preset registry
already covers most actively-used diffusion families:

| Family | LoRA | Full FT | Trainer module |
|---|---|---|---|
| SDXL | ✅ | ✅ | `bracket/trainer/sdxl.py`, `sdxl_full.py` |
| Z-Image | ✅ | ✅ | `bracket/trainer/zimage_lora.py`, `zimage_full.py` |
| Flux-2-Klein 9B | ✅ | — | `bracket/trainer/flux2_klein_lora.py` |
| Flux.1 (dev / schnell) | ✅ | ✅ | `bracket/trainer/flux1_lora.py`, `flux1_full.py` |
| Flux.1-Kontext | ✅ | — | `bracket/trainer/flux1_kontext_lora.py` |
| Qwen-Image 20B | ✅ | ✅ | `bracket/trainer/qwen_image_lora.py`, `qwen_image_full.py` |
| Qwen-Image-Edit | ✅ | — | `bracket/trainer/qwen_image_edit_lora.py` |
| SD3.5 (Medium / Large) | ✅ | ✅ | `bracket/trainer/sd35_lora.py`, `sd35_full.py` |
| HunyuanVideo 13B | ✅ | ✅ | `bracket/trainer/hunyuan_video_lora.py`, `hunyuan_video_full.py` |
| Wan 2.2 | ✅ | ✅ | `bracket/trainer/wan_lora.py`, `wan_full.py` |
| Wan 2.1 | ✅ | — | (uses `wan_lora.py` with `wan_version="2.1"`) |
| LTX-Video | ✅ | — | `bracket/trainer/ltx_video_lora.py` |
| FramePack | ✅ | — | `bracket/trainer/framepack_lora.py` |

Confirm in [`bracket/registry.py`](../../bracket/registry.py) `PRESETS`
tuple before adding work. If the model is listed but the user can't
see it in the UI, the cause is usually a missing `BRACKET_*_PATH`
default — check the FieldSpec rendering, not the registry.

## Step 1: Read the protocol

[`bracket/trainer/base.py`](../../bracket/trainer/base.py) defines:
- `LaunchSpec` — what one training subprocess needs to launch (cmd,
  env, paths).
- `TrainerConfig` — base dataclass. Subclass it for your trainer's
  knobs.
- `Trainer` ABC with required methods: `declare_search_space()`,
  `baseline_config()`, `curated_configs()`, `prepare_run()`,
  `config_from_dict()`, plus optional `session_setup_commands()` for
  trainers needing pre-cache.

## Step 2: Pick a model file to mirror

Pick the existing trainer most similar in shape:

| Adding | Mirror this |
|---|---|
| sd-scripts LoRA module | `sdxl.py` |
| sd-scripts full FT | `sdxl_full.py`, `sd35_full.py` |
| musubi flow-matching LoRA (image) | `zimage_lora.py`, `qwen_image_lora.py`, `flux1_lora.py` |
| musubi flow-matching full FT (image) | `zimage_full.py`, `qwen_image_full.py`, `flux1_full.py` |
| musubi video LoRA | `wan_lora.py`, `hunyuan_video_lora.py`, `ltx_video_lora.py` |
| musubi image-edit LoRA (paired source/target) | `flux1_kontext_lora.py`, `qwen_image_edit_lora.py` |
| Anchor-frame conditioned video LoRA | `framepack_lora.py` |

Copy that file to `bracket/trainer/<your_model>_<lora|full>.py` and
rename the class.

## Step 3: Define the search space

In `declare_search_space()` return a `SearchSpace` keyed by knob name.
Knobs:
- `FloatKnob(low, high, log=True)` — for learning rates
- `IntKnob(low, high)` — for warmup steps, batch sizes if discrete
- `CategoricalKnob(choices=(...))` — for optimiser, scheduler,
  network_dim
- `FixedKnob(value=...)` — for things you've decided not to search

Pin everything that's not loss-bearing. The TPE controller is most
efficient when only the variables that actually affect quality are
exposed.

## Step 4: Write `baseline_config()` and `curated_configs()`

`baseline_config()` returns the user's "what they'd run by hand today" —
the comparison anchor.

`curated_configs()` returns 3-5 known-good configs from docs / community
/ your own production. These run BEFORE the search controller takes
over (warm-start). Each config must validate against the search space —
there's a test for this in `tests/test_trainer_full_and_zimage.py`.
Copy the validation test pattern.

## Step 5: Implement `prepare_run()`

This is where you build the `LaunchSpec`. Rules:

- **Use `make_accelerate_launch_prefix(venv_python, mixed_precision="bf16")`**
  — never call the trainer script directly. On Blackwell GPUs without
  `accelerate launch`, the Accelerator() init takes 2000 s/it instead
  of 2-3 s/it.
- **Use `make_subprocess_env()`** — caps OMP/MKL/OPENBLAS/NUMEXPR
  thread counts. Without these, every DataLoader worker spawns 16 OMP
  threads → catastrophic context switching.
- **Per-run dataset TOML** if the trainer takes batch_size from the
  dataset file (musubi-tuner does, sd-scripts doesn't). Use
  `derive_run_toml()` from `bracket/dataset/runtime.py`.
- **Sample dir path** must match where the trainer writes images. Both
  sd-scripts and musubi write to `<output_dir>/sample/` (singular, not
  `samples/`). Don't change this.
- **Subprocess output never piped through `subprocess.PIPE`** — always
  redirect to a log file directly. Tqdm `\r` updates fill OS pipe
  buffers and freeze the trainer.

## Step 6: Register a preset

Edit [`bracket/registry.py`](../../bracket/registry.py). Add a
`ModelPreset` to the `PRESETS` tuple:

```python
ModelPreset(
    id="<unique-id>",                        # e.g. "hunyuan-lora"
    model_family="<family>",                 # e.g. "HunyuanDiT"
    training_type="<LoRA|Full FT>",
    display_name="<UI label>",               # e.g. "HunyuanDiT · LoRA"
    trainer_factory=_build_<your>,           # ctor wrapper
    fields=(...),                            # FieldSpecs the user fills in
    notes="...",                             # 1-2 sentence note shown in UI
    needs_pre_cache=False,                   # True if trainer needs separate cache step
),
```

If `needs_pre_cache=True`, also implement `session_setup_commands()`
returning the cache-script LaunchSpecs (mirror what `zimage_full.py`
does for `zimage_cache_latents` and
`zimage_cache_text_encoder_outputs`, or `wan_lora.py` for the video
cache scripts).

For trainer-infrastructure paths (musubi-tuner directory, sd-scripts
directory, trainer venv python), reuse `_MUSUBI_DIR_FIELD`,
`_SD_SCRIPTS_FIELD`, `_VENV_PYTHON_FIELD` — they default to the
in-repo `vendor/` submodule paths so users don't type them.

For model-weight paths, define a new `BRACKET_*_PATH` env var and
reference it via `_env_or_default("BRACKET_NEWMODEL_DIT_PATH", "")`.
Add a corresponding commented-out line to the `.env` template in
`install.ps1` and `install.sh` so users know which env vars exist.

## Step 7: Add tests

Create `tests/test_trainer_<your_model>.py`. Use the fake-python +
fake-script pattern from
[`test_trainer_sdxl.py`](../../tests/test_trainer_sdxl.py) — never
launch the real trainer. Cover:

- Search-space shape (knobs present, ranges sensible)
- `baseline_config()` validates against the search space
- `curated_configs()` all validate
- `prepare_run()` builds a sensible cmd (asserts on flag
  presence/absence)

## Step 8: Update README's Supported-trainers table

Add a row to the table in [`README.md`](../../README.md). Mark LoRA /
Full FT support and add a one-line note about the trainer's quirks.

## For LLMs (Axolotl, torchtune, unsloth) — not yet wired

The same protocol works for LLM fine-tuners. Search space changes
(LR ~5e-5, warmup_ratio instead of warmup_steps, optimizers like Lion
or AdamW8bit). Scoring would need an LLM-judge variant of `SampleJudge`
(perplexity? task-eval? structured-output adherence?). See
[`docs/ROADMAP.md`](../../docs/ROADMAP.md) for the LLM expansion plan.

## Anti-patterns — don't do these

- Don't bypass `accelerate launch` "to keep it simple". Performance
  falls off a cliff.
- Don't pipe stdout via `subprocess.PIPE`. Tqdm freezes the trainer;
  write directly to a log file.
- Don't add knobs to the search space that the trainer doesn't accept.
  Validation fails at runtime, not at registration.
- Don't ship hardcoded user-machine paths in `FieldSpec(default=...)`.
  Use `os.environ.get("BRACKET_*_PATH", "")` and document the env var.
- Don't add a new vendor source repo without making it a git submodule
  under `vendor/` — see `docs/UPDATING_TRAINERS.md`.
