<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-dark.png">
    <img src="./assets/logo.png" alt="Bracket" width="160">
  </picture>
</p>

<h1 align="center">Bracket</h1>

<p align="center">
  <em>Train the same diffusion model eight ways. Pick the one that looks best. With a p-value.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/bracket-ml/"><img src="https://img.shields.io/pypi/v/bracket-ml.svg" alt="PyPI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

---

You're fine-tuning Z-Image on 200 portraits. Your last 40k-step run had flat loss
for 38k steps and you still don't know if your learning rate was wrong, your
warmup was wrong, or the dataset is the issue. The Discord said `lr=1e-5`. The
Civitai post said `lr=4e-6`. Loss is too noisy to read. You're about to spend
another eight hours guessing.

Bracket runs the same fine-tune at eight different configurations on a subset of
your data, has a vision model rate the generated samples, and tells you which
config wins — with confidence intervals.

```text
                       config       score   adherence  quality  Δ vs baseline   95% CI         p
   ────────────────────────────────────────────────────────────────────────────────────────────────
1. cand-007 (lr=2e-6)   0.412      8.4/10     8.1/10   −0.187          [−0.24, −0.13]  0.003
2. cand-003 (lr=5e-6)   0.451      8.0/10     7.9/10   −0.148          [−0.21, −0.09]  0.011
3. baseline             0.599      6.8/10     7.2/10    —               —              —
```

It runs against the trainers you already use — `sd-scripts` for SDXL, `musubi-tuner`
for Z-Image and Flux-2-Klein — through real `accelerate launch` subprocesses.
Bracket never re-implements training. It searches.

---

## Install

```bash
# Linux / macOS / WSL2
python -m pip install bracket-ml
```

```powershell
# Windows / PowerShell — drop into your existing trainer venv
& "C:/path/to/your/trainer-venv/Scripts/python.exe" -m pip install bracket-ml
```

Python 3.10+. Requires the trainer (sd-scripts or musubi-tuner) installed
separately — Bracket calls into them, it does not vendor them.

## Quick start

```bash
bracket --trainer zimage-full \
        --dataset-toml ./configs/portraits.toml \
        --sample-prompts ./configs/prompts.txt \
        --budget 8 --seeds-per-config 2 \
        --max-steps-per-run 300 \
        --judge lmstudio \
        --output-dir ./runs/portraits-001
```

Or open the UI:

```bash
bracket-ui   # opens http://127.0.0.1:7860
```

A typical session on a single 5090: budget 8, 2 seeds each, 300 steps per run,
plus a 3-finalist long stage at 1500 steps — about 2 to 3 hours wall clock. Each
trial writes its own `logs/stdout.log` and tfevents under
`runs/<session>/runs/<run_id>/`.

## How it works

```
                            ┌─────────────────────────┐
                            │   bracket orchestrate   │
                            │   stage 1 (short runs)  │
                            └────────────┬────────────┘
       baseline (your hand-tuned config) │
                          ↓              │
       curated known-good warm-start ────┤
                          ↓              │
       Optuna TPE / random search ───────┤  knobs → trainer
                                         │  trainer → samples + tfevents
                                         │  samples → VLM judge (LMStudio + Qwen3-VL)
                                         ↓
                            ┌─────────────────────────┐
                            │  Top-K finalists →      │
                            │  longer-run finals      │
                            └────────────┬────────────┘
                                         ↓
                            ┌─────────────────────────┐
                            │  Markdown report:       │
                            │  Welch's t · 95% CI     │
                            └─────────────────────────┘
```

Five stages:

1. **Baseline.** Your hand-tuned config runs first. Everything else is measured against it.
2. **Curated warm-start.** Each trainer adapter ships a small set of known-good configs (e.g. Adafactor + warmup=50 for Z-Image full-FT). Bracket runs those before anything random.
3. **Search.** Optuna TPE (smart — learns from history) or random (baseline). Each candidate runs `seeds_per_config` times so the score has variance.
4. **Finals.** Top-K candidates by mean score get a longer second-stage run. Catches "looks good at 300 steps but plateaus by 1500".
5. **Report.** Markdown out: ranked configs, sample-quality breakdown, Welch's t-test on best vs runner-up, 95% CI on best vs baseline.

## Why Bracket and not...

- **Optuna alone.** Optuna doesn't know what a diffusion sample is. It will minimise your training loss happily while your samples melt. Bracket uses Optuna *underneath* and adds the visual signal Optuna lacks.
- **W&B Sweeps.** Same blind spot, plus a paywall and a remote dashboard for what should be a local-first tool. Bracket emits all artifacts to a directory you already have.
- **Hand-running sd-scripts / musubi-tuner.** That's exactly what Bracket replaces — and it doesn't replace the trainers themselves, it drives them.
- **AI-Toolkit.** AI-Toolkit is a unified *trainer* with a UI; it doesn't search hyperparameters or judge outputs. Bracket complements it (you could plug AI-Toolkit in as a trainer adapter).
- **Civitai's online trainer.** A black box on someone else's GPU. Bracket runs on your hardware, your data never leaves the box, and you can read the source.

## Supported trainers

| Trainer | LoRA | Full FT | Notes |
|---|:-:|:-:|---|
| SDXL | ✓ | ✓ | via sd-scripts |
| Z-Image (base / Turbo) | ✓ | ✓ | via musubi-tuner; Qwen3 text encoder; auto pre-cache |
| Flux-2-Klein 9B | ✓ | — | via musubi-tuner; Mistral-3-Small text encoder |

Adding a new trainer: implement the `Trainer` protocol in `bracket/trainer/` and register a preset. ~150 lines for SDXL, can be done in an afternoon.

## The judge

By default, Bracket scores runs purely from training loss (cheap, but fooled by overfit-shaped curves). Wire up a local [LMStudio](https://lmstudio.ai) vision model (Qwen3-VL, LLaVA, MiniCPM-V) and Bracket will:

1. Generate sample images via the trainer's own sampling step.
2. Send each image + the prompt that produced it to LMStudio.
3. Get back JSON scores 0–10 on prompt adherence, visual quality, artifact-freeness.
4. Combine with the loss signal (weights configurable; default 0.3 loss / 0.7 sample).

The judge runs locally. Your samples don't leave your machine. The judge is hot-swappable — see [`docs/judges.md`](./docs/judges.md) for adapting it to OpenAI / Claude / a custom local model.

## Confidence claims, honestly

Bracket is rigorous about what it can and can't tell you.

It can tell you:
> Within the declared search space and the budget, configuration **C** had the lowest mean score across **N** seeds. It beat the hand-tuned baseline by **Δ** (95% CI: [low, high]; Welch's t p=**p**).

It cannot tell you:
- That **C** is the globally optimal config — search is bounded by your budget.
- That **C** will still win at 8000 steps. The finals stage mitigates this; it doesn't eliminate it.
- That **C** generalises to a different dataset.

When seeds-per-config is 1, the report says "single-seed: confidence interval skipped". When p-values are noisy, it says "marginal — extend budget". No marketing.

<details>
<summary><strong>Sample report (real output, names changed)</strong></summary>

```markdown
# Bracket · orchestration report

- Training runs: 18 (unique configs: 9, scored: 9, disqualified: 0)
- Multi-seed: up to 2 seeds per config — confidence intervals computed
- Visual scoring: 18/18 runs judged by LMStudio (qwen3-vl-8b)

## Verdict

**cand-007** beat the baseline by Δ=−0.187 (lower is better).
95% CI: [−0.241, −0.133]. Welch's t-test p = 0.003. **High confidence.**

## Top configs

| rank | config_id | role      | n | mean   | std    | learning_rate | warmup | dim |
| 1    | a3f...c1  | candidate | 2 | 0.412  | 0.014  | 2e-6          | 100    | 32  |
| 2    | b07...4e  | candidate | 2 | 0.451  | 0.022  | 5e-6          | 50     | 32  |
| 3    | 11b...a9  | curated   | 2 | 0.488  | 0.018  | 1e-5          | 50     | 16  |
```

</details>

## Roadmap

- [ ] **v0.2** — Per-step VLM scoring (intermediate sampling): catch divergence at step 200 instead of step 500.
- [ ] **v0.2** — React + Vite UI to replace Gradio (in flight; see [`docs/FRONTEND_MIGRATION_PLAN.md`](./docs/FRONTEND_MIGRATION_PLAN.md)).
- [ ] **v0.3** — True ASHA (multi-rung promotion, async — kills bad runs early).
- [ ] **v0.4** — Cross-trainer transfer learning: TPE warmup from prior sessions on related models.
- [ ] **v0.5** — Adapters for AI-Toolkit, simpletuner, OneTrainer.

Not on the roadmap: distributed multi-node, cloud bursting, paid tiers. Bracket is local-first and stays that way.

## Contributing

Issues and PRs welcome. Before opening a PR:

1. `pytest -q` should pass — the suite covers trainer adapters, search, scoring, and the orchestrator loop end-to-end.
2. New trainer adapter? Add a unit test under `tests/test_trainer_<name>.py` modelled on `test_trainer_sdxl.py`.

## License

MIT. See [`LICENSE`](./LICENSE).
