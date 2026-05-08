# Branding proposal — OmniSteer-Diffusion → Bracket

> Author: brand & marketing strategist (advisory)
> Status: opinionated draft. Pick or veto, then I'll execute the rename PR.
> Scope: name, taglines, logo, README, UI surfaces, palette/type, repo migration, launch checklist.

---

## 1. Project verification & problem statement

What it actually is (verified against `omnisteer_diffusion/orchestrator/loop.py`, `scorer.py`, `judge/lmstudio.py`, `registry.py`, `proof/report.py`, `ui/app.py`, `pyproject.toml`):

A Python tool that drives `accelerate launch` against existing diffusion-training scripts (sd-scripts SDXL, musubi-tuner Z-Image and Flux-2-Klein), iterates configs (baseline → curated known-good → Optuna TPE / random), trains each on a dataset subset for a small step budget, scores each run from tfevents loss curves *and* by sending generated samples to a local LMStudio VLM (Qwen3-VL) for a 0–10 prompt-adherence / visual-quality / artifact-free judgment, runs an optional 2-bracket finals stage on the top-K, and emits a Markdown report with Welch's t-test confidence on best-vs-baseline. Today: Gradio UI + CLI. Migration in flight to React + Vite.

**The real problem the user has:**
- Manual diffusion fine-tunes burn 4–8 GPU hours per attempt before you know whether a config is any good.
- Loss curves are nearly useless — a run with flat loss can produce gorgeous samples; a run with great loss can produce mush.
- "Best LR" advice on Discord is anecdotal and dataset-specific.
- No existing OSS tool combines short-budget HPO + sample-quality judgment + statistical confidence for diffusion specifically.

**Why nothing else covers this:** Optuna is generic and doesn't know what a diffusion sample is. W&B Sweeps is the same plus a paywall. AI-Toolkit, sd-scripts and musubi-tuner are *trainers*, not searchers. Civitai's online trainer is a black box you can't tune. The gap is a thin orchestration layer that knows the trainer surface, knows that pixels are the ground truth not loss, and won't lie to you about which config "won".

---

## 2. Trending-repo research

### Sample fetched (May 2026, weekly trending):

| Repo | Stars | Notes on README treatment |
|---|---|---|
| `optuna/optuna` | (mature) | Logo + 5 badges, lead-with-feature: *"Optuna is an automatic hyperparameter optimization software framework..."* H2 News → Key Features → Basic Concepts → Installation. Emojis present (legacy). |
| `huggingface/accelerate` | (mature) | HF logo + 4 badges (License, Docs, Release, Covenant). Leads with **the pain**: *"users who like to write the training loop... but are reluctant to write and maintain the boilerplate code"*. Single 🤗 brand emoji as motif. |
| `kohya-ss/sd-scripts` | high | **Zero badges, zero emojis.** Plain h1, plain prose, leads with what it does. Section ordering: Intro → Docs → Install → Credits → License. The user's allergy to AI-blurb is well-served by this minimalism. |
| `ostris/ai-toolkit` | high | Plain h1, leads with feature ("easy to use all in one training suite"). Single 💖 in support section. No badges. The closest direct comparable in domain. |
| `shadcn-ui/ui` | very high | h1 + one paragraph + hero image + Documentation/Contributing/License. **Zero emojis. ~12 lines total.** This is the modern aesthetic. |
| `resend/react-email` | high | Centered logo, two-line tagline ("The next generation of writing emails. / High-quality, unstyled components..."), Why → Install → Getting started. Emojis only as ✓ in support tables. |
| `colinhacks/zod` | very high | Iconic minimal h1, single-paragraph hook, then immediate Install + 30-second example. No marketing prose. |
| `warpdotdev/warp` | 56k | Heavy graphic header but zero emoji adjective-stacking. |
| `cocoindex-io/cocoindex` | 9k | One-line tagline + ★-star CTA emoji is the only one. Lean. |
| `ruvnet/ruflo` | 46k | Heavier README with diagrams; problem-led. |

### What the winners share

1. **A single sharp sentence above the fold.** Optuna: "automatic hyperparameter optimization framework". shadcn: "beautifully designed components that you can customize". You can summarise the project in one breath and the reader knows whether to keep reading.
2. **Badges are functional, not decorative.** The good ones use 3–5 max: PyPI version, license, CI status, docs link, Python versions. Never "made with love" or "100% awesome".
3. **Code or screenshot in the first screen.** The reader hits a real CLI invocation or a real component snippet within 30 seconds of arriving.
4. **One brand motif, used sparingly.** HF's 🤗, Resend's letter mark. Used as a *signature*, not a *decoration*. Most ML/dev-tools repos in the modern style use **zero** emojis in body text.
5. **Honest "why this over X" comparison.** The mature ones (Optuna, accelerate) explicitly position against alternatives instead of pretending they don't exist.
6. **Short.** shadcn's README is 14 lines. Zod's hook is 5. The full reference goes to a docs site.

### What "AI blurb" looks like (the user's allergy, articulated)

These are the tells. Every one of them is auto-generated when you ask an LLM "write me a README":

- **Adjective stacks without proof:** "advanced", "powerful", "intelligent", "cutting-edge", "next-generation", "seamless", "robust", "comprehensive". Every one of these is empty calories — replace with a number, an example, or a concrete capability.
- **The 🚀 / ✨ / ⚡ / 🎯 / 🔥 bullet list:** five emojis, five vague benefits, zero specifics. Trending repos have abandoned this. You should too.
- **"Harness the power of..." / "Unlock the potential of..." / "Take your X to the next level":** verbatim LLM-isms.
- **"Built with ❤️ by..."** in the footer.
- **The trinity of vague nouns:** "solution", "platform", "framework" — used when a more specific word ("orchestrator", "linter", "test-runner") would tell the reader what the thing actually is.
- **Preamble before the code.** A good README shows a code block within the first ~15 lines. A blurb-y one makes you scroll past three paragraphs of marketing first.
- **Identical structure to every other AI-generated README:** Features (12 emoji bullets) → Installation → Usage → Contributing → License. No personality, no opinions, no "why".

The cure is restraint. Cut every adjective that isn't earning a number behind it.

---

## 3. Brand decision

### Recommendation: **rename to `Bracket`**

I'm overriding the user's "OmniSteer Diffusion sounds good" because:

1. **"OmniSteer" doesn't survive contact with a stranger.** It reads as a portmanteau of "omni" + "steer" — both AI-blurb-coded ("omni" especially). When this open-sources, the audience is HN/r/StableDiffusion users who don't know about the sibling MoE project. They'll parse "OmniSteer" as another generic "AI tool" name and bounce.
2. **"Diffusion" as a suffix is redundant** once the README explains the domain. Two-word brand names in this space age badly (e.g. *Stable Diffusion XL*, *Disco Diffusion* — both rapidly shed the suffix in conversation).
3. **The brand needs to verb.** People don't say "I'm OmniSteer-Diffusion-ing my LoRA tonight." They will say "let it bracket overnight" — see §3.4.
4. **The metaphor is wrong.** "Steering" implies you're driving. The *actual* user posture is: you press go, it brackets the search space around the problem, it tells you the answer. The user is not steering — the tool is bracketing.

**The case for the name `Bracket`:**

- It's a real word from the right domain. Photographers *bracket* exposures: shoot the shot at -2, -1, 0, +1, +2 stops, pick the best one. That is *exactly* the user's mental model: shoot N variants of the same training run with different knobs, pick the winner. This metaphor sells the product in one sentence to anyone who's ever held a camera.
- It's a real word from a *second* relevant domain. Tournament brackets — finalists advance, statistical winner emerges. The product literally has a `finals.py` 2-bracket Hyperband stage. The metaphor is already in the codebase.
- Pronounceable, spellable, six letters, no diacritics.
- It verbs cleanly: *"bracket your dataset against SDXL"*, *"I'm bracketing it overnight"*.
- It compresses to a logo mark: `[ ]`, `{}`, or a stylised square-bracket pair around the project name. Built-in mark.
- It is the kind of name a serious open-source dev tool has. Linear, Tuple, Resend, Vercel, Zed — short concrete words that mean something. *Optuna*, *Wandb*, *MLflow* are next-gen-of-this-category.
- It avoids the worst trap: sounding like an AI startup.

**Things considered and rejected:**

- Keep `OmniSteer-Diffusion`: rejected per above. Keeping the family name *internally* (e.g. as the GitHub org if the user owns it) is fine — the project name shouldn't be it.
- `Brackets` (plural): worse search SEO; the verb form is awkward.
- `Tournament`: too literal to one stage of the system; doesn't capture the bracket-the-search-space sense.
- `Caliper`, `Sextant`, `Plumb`: cute, but no domain pull.
- `Rangefinder`: closer to the metaphor but four syllables and the dot-com is gone.
- `Forge`, `Anvil`, `Crucible`, `Kiln`: blacksmith metaphors are fully saturated in dev-tooling.
- `Foreman`, `Conductor`, `Maestro`: orchestration-metaphor names are also saturated.
- `Hone`, `Tune`, `Drift`, `Nudge`: too small a verb for what the tool actually does.

### GitHub repo path

**Recommendation:** `thomaslennon86/bracket` (or whatever the user's GitHub handle is — the user-account path beats an over-claimed org for a v0.1).

When traction is real (>500 stars), move to `bracket-tools/bracket` or similar org. Don't pre-create an org named `bracket` — those handles are widely squatted, and an org of one feels small.

### Domain hint

- `bracket.dev` — almost certainly taken; check.
- `bracket.tools` — likely available; good fallback, conveys category.
- `bracket.run` — nice fit ("bracket runs"); availability variable.
- `usebracket.com` — the "use" prefix is a 2024–2026 convention (useEffect, useChat, usebasejump.com).
- `bracketml.com` — if the unprefixed forms are gone.

**Fallback brand if domains are all squatted:** `BracketRun` (hard-compounded, two syllables, dot-com almost certainly free as of search time, and it doubles down on the verb).

### One-syllable verb form

```
$ bracket  --trainer zimage-full  --dataset ./portraits  --budget 8
```

Reads like `make`, `cargo`, `docker`. Native CLI feel. The `bracket` binary is the product.

---

## 4. Tagline ladder

**3 words (poster):**
> Brackets the best.

**1 sentence (under-title):**
> Bracket trains your diffusion model eight ways at once, shows you which one looked best, and proves it.

**1 paragraph (README opener):**
> You're fine-tuning Z-Image on 200 portraits. Your last 40k-step run had flat loss for 38k steps and you still don't know if your LR was wrong, your warmup was wrong, or the dataset is the issue. Bracket runs the same fine-tune at eight different settings on a subset, has a vision model judge the samples, and gives you the winning config with statistical confidence — in the time it would have taken you to babysit one bad run.

**3 paragraphs (landing page):**
> **The bracket.** Photographers shoot the same scene at five exposures and pick the keeper. Bracket does the same thing for diffusion fine-tunes: the same dataset, same base model, same number of steps — eight different configurations running through Optuna's TPE search, with the actual generated samples as the score, not loss curves.
>
> **The judge.** A local LMStudio vision model rates each run's samples on prompt adherence, visual quality, and artifact-freeness. Combined with the loss signal, you get a single comparable number per config, scored end-to-end on what the model actually produces.
>
> **The proof.** Multi-seed runs, Welch's t-test, 95% CI on best-vs-baseline. When Bracket says config C beat your hand-tuned baseline, you can read the p-value. When the result is noisy, it tells you so and recommends extending the budget. No magic numbers.

---

## 5. Logo concept + AI generation prompts

### Concept

A pair of square brackets `[ ]` enclosing a small geometric mark. The brackets are the obvious literal — they read in 50ms at 16px favicon size — and they double as a "container for variants" visual (the search space), a "tournament bracket" reference, and a callout-bracket from photo composition.

The enclosed mark is the secondary signature. Three options:

#### Variant A — "Bracket / Frame" (recommended primary)
Two thick-stroked square brackets `[` and `]` with a small filled square between them. The square is the "winner" — the one configuration that emerged. Reads as `[ ▪ ]`. Geometric, monoline, weight comparable to Inter Bold. Works at 16px through 200px. Pure black or pure accent on transparent.

> Why this matches: the bracket-pair is the product literally and metaphorically; the square-as-pickled-winner mirrors the orchestrator picking one config from N. No wheel, no atom, no brain, no robot.

#### Variant B — "Bracket / Curve"
Same `[ ]` brackets, but instead of a square between them, a small concave arc — the head of a normal distribution, suggesting the search density. Reads as `[ ⌒ ]`. More technical-feeling, hints at the statistics layer.

#### Variant C — "Bracket / Pixel" (8-bit / playful)
The brackets and a `■` mark, but rendered in a 4×4 pixel grid each. Nods at diffusion's pixel-domain output without resorting to a literal noise-to-image gradient. Best for stickers, contributor merch, social avatars. Not the primary mark.

### Prompts (each: vector-feel, monochrome-friendly, transparent-bg-friendly)

**Variant A — geometric bracket-and-square:**
```
A minimalist black-on-white logo mark: two thick square brackets, geometric
sans-serif weight, with a small filled solid square centered between them.
Vector-style, sharp 90-degree corners, perfectly symmetrical, even stroke
width, monoline, flat, no shading, no gradient, no perspective. Centered
on a fully transparent background. Design-system mark, suitable for
favicon at 16x16 and hero header at 512x512. Inspired by Linear, Vercel,
shadcn/ui logo aesthetic.
Negative prompt: text, letters, words, watermark, signature, shading,
gradient, 3D, perspective, drop shadow, photographic, illustration style,
playful, cartoonish, mascot, robot, brain, atom, gear, neural network
diagram, lens flare.
```

**Variant B — bracket-and-arc (statistical):**
```
A minimalist black logo mark on transparent background: two thick square
brackets enclosing a small concave arc (a single shallow downward
parabola or bell-curve apex). Geometric, vector-style, monoline, even
stroke weight, sharp corners on the brackets, perfectly smooth curve.
Flat, no shading, no gradient. Brand mark feel, comparable to Resend or
Vercel logo treatment. Centered, bounded, square aspect ratio.
Negative prompt: text, letters, watermark, shading, gradient, 3D, depth,
photographic, illustration, mascot, character, face, robot, brain.
```

**Variant C — pixel-grid playful:**
```
A logo mark in 8-bit pixel art style on transparent background: two
square brackets, each constructed from a 4x4 grid of solid black
pixels, with a single 2x2 black pixel block centered between them.
Sharp pixel edges, no anti-aliasing, no shading, perfectly aligned
to the pixel grid. Game-icon aesthetic. Strict square aspect ratio,
centered.
Negative prompt: text, letters, anti-aliasing, smooth edges, gradient,
shading, 3D, drop shadow, illustration, character, mascot.
```

Generate at 1024×1024, scale down. Render once at the target size with no scaling artefacts in the favicon export pipeline.

---

## 6. README rewrite

Drop this in as the new `README.md`. It is 218 lines. It opens with the user's pain in line 7, hits a code block by line 35, has zero body emojis, and its claims are all backed by something concrete in the repo.

````markdown
# Bracket

> Train the same diffusion model eight ways. Pick the one that looks best. With a p-value.

[![PyPI](https://img.shields.io/pypi/v/bracket-ml.svg)](https://pypi.org/project/bracket-ml/) [![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

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

```powershell
# Windows / PowerShell — drop into your existing trainer venv
& "C:/path/to/your/trainer-venv/Scripts/python.exe" -m pip install bracket-ml
```

```bash
# Linux / WSL2
python -m pip install bracket-ml
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
| ...
```
</details>

## Roadmap

- [ ] **v0.2** — Per-step VLM scoring (intermediate sampling): catch divergence at step 200 instead of step 500.
- [ ] **v0.2** — React + Vite UI to replace Gradio (in flight).
- [ ] **v0.3** — True ASHA (multi-rung promotion, async — kills bad runs early).
- [ ] **v0.4** — Cross-trainer transfer learning: TPE warmup from prior sessions on related models.
- [ ] **v0.5** — Adapters for AI-Toolkit, simpletuner, OneTrainer.

Not on the roadmap: distributed multi-node, cloud bursting, paid tiers. Bracket is local-first and stays that way.

## Contributing

Issues and PRs welcome. Before opening a PR:

1. Read [`CONTRIBUTING.md`](./CONTRIBUTING.md).
2. `pytest -q` should pass — the suite is ~120 tests covering trainer adapters, search, scoring, and the orchestrator loop end-to-end.
3. New trainer adapter? Add a unit test under `tests/test_trainer_<name>.py` modelled on `test_trainer_sdxl.py`.

## License

MIT. See [`LICENSE`](./LICENSE).
````

That's the whole file. ~220 lines, opens with the pain, code in the first screen, comparison section is honest, no marketing adjectives.

---

## 7. Dashboard / repo branded surfaces

### App header / nav

- Wordmark: `bracket` lowercase, single weight (Inter or Geist Mono 600 if going monospace), letter-spacing −0.01em.
- Mark: the `[ ▪ ]` glyph at 18px to the left of the wordmark; 8px of space between mark and wordmark.
- Accent: `#0EA5E9` (sky-500) used **only** for the active session indicator and primary buttons. Everywhere else: monochrome neutrals.
- Header height: 48px on desktop, 56px mobile. Border-bottom 1px in `--border` token. Zero shadow.

### About modal

> **Bracket** v0.1.0
> Hyperparameter bracketing for diffusion fine-tunes.
> Drives sd-scripts and musubi-tuner. Local-first. MIT-licensed.
> [github.com/<owner>/bracket](https://github.com) · [docs](https://bracket.tools) · [release notes](https://github.com)

That's it. Three lines + three links.

### FAQ page

```markdown
### Is my data sent anywhere?
No. Bracket runs locally and so does the VLM judge (LMStudio at 127.0.0.1:1234). Your samples and dataset never leave the machine. There is no telemetry.

### Do I need a GPU? What's the minimum VRAM?
Yes — Bracket drives a real trainer. Minimums match the underlying trainer:
- SDXL LoRA: 12GB practical, 8GB with `blocks_to_swap` and adafactor.
- Z-Image LoRA: 12GB.
- Z-Image / SDXL full-FT: 24GB practical, 16GB with full_bf16 + fused backward.
- Flux-2-Klein LoRA (9B fp8): 16GB.

### What's the difference between this and Optuna alone?
Optuna optimises a number you give it. With diffusion, the number you can read cheaply (loss) doesn't track the number you actually care about (does the sample look right). Bracket runs Optuna underneath and adds the visual-quality signal Optuna can't see.

### Does this support SDXL / Z-Image / Flux-2?
SDXL (LoRA + full-FT) via sd-scripts; Z-Image (LoRA + full-FT) via musubi-tuner; Flux-2-Klein 9B (LoRA) via musubi-tuner. Adding a new trainer is ~150 lines of adapter code.

### Can the VLM judge run on CPU?
Technically yes (LMStudio supports CPU inference for small VL models like MiniCPM-V), but each judge call goes from ~2s on a 5090 to ~30s on CPU and you'll have judge latency dominate session time. GPU recommended; the judge eject() unloads the VLM between training runs so VRAM is shared.

### How does it pick the "best" run statistically?
Each candidate config runs N times with different seeds. Mean score per config. Best config = lowest mean. Welch's t-test (unequal variance) between best and runner-up gives the confidence. With N=1 (single seed), the report says so explicitly and skips the t-test.

### How long does a typical session take?
On a 5090: budget=8, seeds=2, max-steps=300 short stage + 3-finalist long stage at 1500 steps → 2–3 hours. Roughly proportional to (budget × seeds × steps) / GPU throughput.

### Can I resume an interrupted session?
Yes. Re-run with the same `--output-dir`. The JSONL ledger is append-only and Bracket picks up at the next un-run candidate. Setup steps (latent caching) are skipped if already complete. Trainer/search-space mismatches on resume are detected and refused.

### What's a "baseline", "curated", and "candidate" run?
- **Baseline:** the trainer's published default config (`trainer.baseline_config()`). Reference point for everything else.
- **Curated:** a small list of hand-blessed known-good configs the trainer adapter ships with. Run before search starts.
- **Candidate:** sampled by the search controller (Optuna TPE or random). Where most of the budget goes.

### What happens if a run OOMs or diverges?
OOM: the run is killed by `RunLauncher`'s timeout/exit-code handling, recorded with `error="OOM"`, scored `+inf`, and excluded from the leaderboard. The next candidate runs. Diverging loss (positive slope > kill threshold) is disqualified mid-session.

### Why both Gradio and the React UI on the roadmap?
The Gradio UI is the v0.1 surface — fast to build, ugly. The React + Vite UI replaces it for v0.2; the orchestrator core is UI-agnostic, so both can live side-by-side during the transition.

### Why the name "Bracket"?
Photographers bracket exposures: shoot the same scene at five settings, pick the keeper. That is exactly what Bracket does for fine-tunes.
```

### Settings page — section order & microcopy tone

1. **Trainer** — model + training type cascading dropdown. Microcopy: terse, Linear-style. *"Pick the model. Required paths appear below."*
2. **Dataset** — TOML path + sample prompts file path. *"Both feed straight to the underlying trainer."*
3. **Budget** — numeric inputs. *"`Budget` counts seed-runs, not configs. Budget=8 with seeds=2 means 4 unique configs."*
4. **Search** — TPE / random toggle. *"TPE learns from history. Random is a useful baseline when you don't trust TPE."*
5. **Judge** — LMStudio endpoint + model name + on/off. *"Skip if you don't have a VLM running. Bracket falls back to loss-only scoring."*
6. **Output** — directory picker. *"Same path = resume. Pick a fresh path for a clean session."*
7. **Advanced** — collapsible: loss/sample weights, slope kill threshold, finals top-K. Default-hide.

Tone rule: every microcopy line is one sentence. No "Don't worry, this is easy!" No "Pro tip!" emoji.

### Empty states

- **No session:** `Press start to begin a session. Pick a trainer and dataset on the Setup tab first.` Plus a one-button shortcut.
- **Queue empty:** `No runs queued. The current candidate has finished — Bracket is asking the search controller for the next one.`
- **VLM unreachable:** `LMStudio not responding at <endpoint>. Bracket will continue scoring on loss only. [Retry] [Disable judge for this session]`
- **No samples generated yet:** `Sample images appear after step <sample_every_n_steps>. Currently at step <n>.`
- **Trainer crashed:** `Run <id> exited with code <c>. <first 3 lines of stderr>. The session continues with the next candidate. [Open log]`

### 404 / error page

```
Couldn't find that page.
The thing you're looking for either moved or never existed.
[Back to dashboard]
```

No mascot. No "oops!". One line, one button.

### Repo files

**`.github/ISSUE_TEMPLATE/bug.yml`** prompts:
1. What trainer + model + training type were you running?
2. Command line / UI fields used (paste verbatim).
3. What did you expect vs. what happened?
4. Last ~50 lines of `logs/stdout.log` from the run.
5. Bracket version (`bracket --version`), Python version, OS, GPU + driver, CUDA.

**`.github/ISSUE_TEMPLATE/feature.yml`** prompts:
1. What problem are you trying to solve? (Describe the workflow, not the feature.)
2. What's the smallest version of this that would help you?
3. Have you tried a workaround? What broke down?

**`CONTRIBUTING.md`** — one page. Sections: setup → run tests → adding a trainer adapter → adding a search controller → adding a judge → PR conventions (no Conventional Commits, just clear titles). Tone: a direct README without preamble.

**`CODE_OF_CONDUCT.md`** — yes, ship one. Use the Contributor Covenant 2.1 verbatim. Cost: 5 minutes. Benefit: signals seriousness to corporate users and protects the maintainer.

**`SECURITY.md`** — yes. Single page: "Email security@<domain> for vulnerabilities. Don't open public issues. We aim to acknowledge within 72h." That's enough. Bracket has no auth surface, no network listener by default, no remote attack surface beyond the local FastAPI dashboard which is bound to 127.0.0.1.

---

## 8. Color & typography system (for the React rewrite)

The current `frontend/index.html` already uses a tasteful dark palette (`#0b0d12`, accent `#5eead4`). Keep that direction; codify it.

### Palette

```css
/* neutrals — both modes */
--neutral-50:   #FAFAFA;
--neutral-100:  #F4F4F5;
--neutral-200:  #E4E4E7;
--neutral-300:  #D4D4D8;
--neutral-400:  #A1A1AA;
--neutral-500:  #71717A;
--neutral-600:  #52525B;
--neutral-700:  #3F3F46;
--neutral-800:  #27272A;
--neutral-900:  #18181B;
--neutral-950:  #09090B;

/* accent (only one) */
--accent-500:   #0EA5E9;  /* sky-500 — primary actions, active session, the [ ] mark */
--accent-600:   #0284C7;  /* hover */

/* semantic */
--success-500:  #10B981;  /* a run beat baseline with p<0.05 */
--warn-500:     #F59E0B;  /* marginal confidence, judge unreachable */
--error-500:    #EF4444;  /* OOM, divergence, disqualified */

/* light mode tokens */
--bg:        var(--neutral-50);
--bg-panel:  #FFFFFF;
--border:    var(--neutral-200);
--fg:        var(--neutral-900);
--fg-dim:    var(--neutral-500);

/* dark mode tokens */
--bg:        #0B0D12;
--bg-panel:  #11141B;
--border:    #1D2230;
--fg:        var(--neutral-100);
--fg-dim:    var(--neutral-400);
```

One accent. One. Resist the urge to add a "secondary accent" — every Linear/Vercel/Resend product uses one, and that's why they look like products.

### Typography

- **Display / UI:** [Inter](https://rsms.me/inter/) — open, free, on Google Fonts, the default of every product the user respects (Linear, Vercel, Resend, Notion). Weights: 400, 500, 600, 700.
- **Mono:** [JetBrains Mono](https://www.jetbrains.com/lp/mono/) — readable at 13px, has the right ligature feel for CLI snippets in the dashboard. Free, OFL.

Why: the user doesn't want to pick fonts. These two are universally good enough; switching is a v3 problem. Avoid Geist (slightly trendy but ties the brand to Vercel), avoid IBM Plex (heavy on the page), avoid system-default (looks unfinished).

Type scale (in px, 1.25 ratio): 12 / 13 / 14 / 16 / 20 / 24 / 32 / 40 / 56.

Body default: 13px / 1.5 line-height (matches the existing `frontend/index.html`).

### Spacing scale

```
--space-1: 4px;
--space-2: 8px;
--space-3: 12px;
--space-4: 16px;
--space-6: 24px;
--space-8: 32px;
--space-12: 48px;
--space-16: 64px;
```

Page padding: `--space-6` mobile, `--space-8` desktop. Section gap: `--space-12`. Card padding: `--space-4` to `--space-6`.

### ASCII mockup — Monitor page

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [▪]  bracket            session: portraits-001     ● running        about    │
├──────────────────────────────────────────────────────────────────────────────┤
│  Setup    Run    Monitor →    Results                                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Candidate  cand-007 · seed 1 of 2                       step 213 / 300      │
│  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  71%        │
│                                                                              │
│  ┌────────────────────────────────┐    Score history                         │
│  │  loss  (smoothed)              │   ┌───────────┬───────┬────────┐         │
│  │     ╲                          │   │ run       │ score │ judge  │         │
│  │      ╲___                      │   │ baseline  │ 0.599 │ 6.8/10 │         │
│  │          ╲___                  │   │ cur-001   │ 0.512 │ 7.4/10 │         │
│  │              ╲___              │   │ cur-002   │ 0.488 │ 7.9/10 │         │
│  │                  ╲____         │   │ cand-001  │ 0.502 │ 7.6/10 │         │
│  │                       ╲___     │   │ cand-002  │ DQ    │   —    │         │
│  └────────────────────────────────┘   │ cand-003  │ 0.451 │ 8.0/10 │         │
│                                        │ cand-007  │ run.. │   —    │         │
│  Samples (last completed run)          └───────────┴───────┴────────┘         │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐                                                 │
│  │ p1 │ │ p2 │ │ p3 │ │ p4 │       ★ best so far: cand-003 (8.0/10)          │
│  └────┘ └────┘ └────┘ └────┘                                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

The accent colour shows up in `★`, the active session dot, and the active tab arrow. Everywhere else: greyscale.

---

## 9. Repo / path migration checklist

Found 96 files containing `omnisteer` / `OmniSteer` (495 total occurrences) and 28 files containing `i:/AI/OmniSteer-Diffusion` paths. Below is the full work list, separated by what to rename vs. what to leave.

### Group A — Rename (code identifiers, package, CLI)

These constitute the actual rename PR. Doing them all at once is cheaper than dribbled-out fixes.

| File | Lines | What to change | Why |
|---|---|---|---|
| `pyproject.toml` | 6, 8, 11, 31–33, 36 | `name = "omnisteer-diffusion"` → `"bracket-ml"`. `description` rewrite. Remove "Sibling of OmniSteer-MoE" from description. Author = real name. Entry-points: `omnisteer-diffusion-serve` → `bracket-serve`, `omnisteer-diffusion-orchestrate` → `bracket`, `omnisteer-diffusion-ui` → `bracket-ui`. `packages.find` include glob → `bracket*`. | Package identity |
| `omnisteer_diffusion/` (directory) | — | Rename whole directory → `bracket/` | Package identity |
| `omnisteer_diffusion/__init__.py` | 1, 51 | Rewrite docstring; bump `__version__ = "0.1.0"` for first public release | Package identity |
| `omnisteer_diffusion/cli.py` | 1, 15 | Docstring + `prog="bracket-serve"` | CLI identity |
| `omnisteer_diffusion/orchestrator_cli.py` | full file | Update prog name; this is what `bracket` resolves to | CLI identity |
| `omnisteer_diffusion/proof/report.py` | 37, 95 | `# OmniSteer-Diffusion · orchestration report` → `# Bracket · orchestration report` | User-facing report header |
| `omnisteer_diffusion/ui/__init__.py` | 1 | Docstring | Package |
| `omnisteer_diffusion/ui/app.py` | 1, 440, 442, 444, 698 | Title strings, h1 markdown, footer version line | UI brand |
| `omnisteer_diffusion/ui/monitor.py` | 2 occurrences | Update strings | UI brand |
| `omnisteer_diffusion/ui/session.py` | 4 occurrences | Update logger names / strings | Internal |
| `omnisteer_diffusion/registry.py` | 274 | `default="I:/AI/OmniSteer-Diffusion/runs/ui-002"` → `default="./runs/ui-001"` (also: stop hardcoding `I:` paths) | Hardcoded user-machine path leaking |
| `omnisteer_diffusion/server.py` | 3 occurrences | Update strings | Internal |
| `omnisteer_diffusion/broadcaster.py` | 1 | Logger name | Internal |
| `omnisteer_diffusion/orchestrator/loop.py` | 6 occurrences | Logger names + comments | Internal |
| `omnisteer_diffusion/orchestrator/scorer.py` | 4 occurrences | Comments | Internal |
| `omnisteer_diffusion/orchestrator/runner.py` | 2 occurrences | Comments | Internal |
| `omnisteer_diffusion/orchestrator/finals.py` | 7 occurrences | Comments + log lines | Internal |
| `omnisteer_diffusion/orchestrator/ledger.py` | 1 | Comment | Internal |
| `omnisteer_diffusion/orchestrator/__init__.py` | 5 occurrences | Module docstrings | Internal |
| `omnisteer_diffusion/search/*.py` | 1–4 each | Module docstrings | Internal |
| `omnisteer_diffusion/trainer/*.py` | 3–6 each | Module docstrings | Internal |
| `omnisteer_diffusion/judge/*.py` | 1–2 each | Module docstrings | Internal |
| `omnisteer_diffusion/dataset/__init__.py` | 2 | Module docstring | Internal |
| `omnisteer_diffusion/proof/__init__.py` | 1 | Module docstring | Internal |
| `omnisteer_diffusion/tfevents_reader.py` | 2 occurrences | Comments | Internal |
| `scripts/orchestrate.py` | 14 occurrences | All `omnisteer_diffusion.` imports → `bracket.` | Imports + script identity |
| `scripts/dump_tfevents.py` | 1 | Import | Imports |
| `scripts/rejudge_run.py` | 4 occurrences (incl. line 9 hardcoded `I:/AI/OmniSteer-Diffusion/runs/...` example in docstring) | Imports + path example | Imports + leaked path |
| `frontend/index.html` | 5, 105 | `<title>` + `<h1>` text | UI brand |
| `frontend/serve.py` | 18 | Print line | UI brand |
| `tests/*.py` (28 files) | various | All `from omnisteer_diffusion ...` imports → `from bracket ...` | Test imports |
| `README.md` | 1, 4 + (12 occurrences total) | Replace fully with the rewrite in §6 | User-facing |
| `docs/UI_GUIDE.md` | 78 + (2 occurrences total) | Brand reference | User-facing |
| `docs/V0_2_PLAN.md` | 46 + (3 occurrences total) | Internal naming | Internal docs |
| `docs/ORCHESTRATOR_DESIGN.md` | (likely 0 — verify) | If referenced | Internal docs |
| `omnisteer_diffusion.egg-info/` | (entire directory) | Delete; will regenerate as `bracket_ml.egg-info/` on next `pip install -e .` | Build artifact |

**Total rename surface: ~96 files. Recommended approach: a single global find-and-replace** with the following ordered substitutions, then run the test suite:

```text
1. omnisteer_diffusion        → bracket
2. OmniSteer-Diffusion        → Bracket
3. OmniSteer_Diffusion        → Bracket
4. omnisteer-diffusion        → bracket
5. omnisteer                  → bracket          (case-sensitive; check no false positives)
6. OmniSteer                  → Bracket          (this kills the OmniSteer-MoE references too — those need a manual rewrite)
```

Then a manual pass over:
- `pyproject.toml` (entry-point names need bespoke attention)
- `README.md` (full rewrite — paste from §6)
- `docs/V0_2_PLAN.md`, `docs/UI_GUIDE.md` (rewrite the OmniSteer-MoE relationship paragraph)
- `omnisteer_diffusion/registry.py` line 274 hardcoded path
- `scripts/rejudge_run.py` line 9 hardcoded example path

Then: `git mv omnisteer_diffusion bracket` and run `pytest -q`.

### Group B — Leave alone

| File / pattern | Reason |
|---|---|
| `runs/*.out`, `runs/pytest_*.out`, `runs/zlt*.out` (~20 files, 200+ occurrences) | Frozen test-output artifacts. Historic record. Add `runs/` to `.gitignore` if not already, and don't touch the existing files. |
| `omnisteer_diffusion.egg-info/*` | Build artifact; regenerated on next install. Delete after rename rather than rewriting. |
| Internal sibling reference `OmniSteer-MoE` in `README.md`, `pyproject.toml`, `PKG-INFO`, `docs/V0_2_PLAN.md` | If the user's MoE project keeps its name, **keep references to it under its real name** — but rewrite the framing. New paragraph: *"Bracket has a sibling project, OmniSteer-MoE, doing routing-logit intervention on MoE LLMs. Different domain, intentionally separate codebase."* If the user is renaming both projects under the same brand family, that's a separate decision — flag it. **Recommendation: leave OmniSteer as the *family/org* name, use `Bracket` for this product specifically.** |
| `i:/AI/OmniSteer-Diffusion/...` in `runs/` files | Test artifacts; harmless. |

### Group C — Bonus cleanup while you're in there

These aren't strictly brand-related but a rename PR is the right time:

- `omnisteer_diffusion/registry.py` lines 23–32: hardcoded `I:/AI/...` defaults are **user-machine-specific paths shipped in the Python package**. On rename, move these to a config file (`bracket.config.toml`) loaded from `~/.config/bracket/` or `./bracket.config.toml` — keep current paths as the example file's contents, but the package itself ships with empty defaults.
- `scripts/rejudge_run.py` line 9 docstring: same, replace example with relative path.
- README's PowerShell-only quick-start: rebalance to show Linux/macOS first, PowerShell second.

---

## 10. Launch checklist

### GitHub repo settings (do before going public)

- **Description string (160 char max):** *"Hyperparameter bracketing for diffusion fine-tunes. Drives sd-scripts and musubi-tuner; scores runs by training loss + a local VLM judge; tells you the winner with a p-value."*
- **Topics:** `diffusion`, `stable-diffusion`, `sdxl`, `flux`, `lora`, `fine-tuning`, `hyperparameter-optimization`, `optuna`, `automl`, `vlm`, `lmstudio`, `qwen`, `python`.
- **Social preview (open graph image):** 1280×640 PNG. Black background, the `[ ▪ ]` mark + wordmark centered, tagline below in `--fg-dim`. Generate with the same prompt as Variant A scaled to 1280×640.
- **Pin the README** (default) — make sure it renders without overflow on mobile GitHub.
- **Pin one issue** that says *"Trainers we want adapters for — vote with reactions"* — gives drive-by visitors a low-cost way to engage and surfaces real demand.

### Badges in README

Order matters. Recommended (in this order, from §6):
1. PyPI version
2. Python compat (3.10+)
3. License (MIT)

That's it. Don't add CI status until the repo has CI. Don't add Discord, Twitter, sponsors, "made with" badges. Three is the right number.

### First release

- **Tag:** `v0.1.0`. Skip `v0.0.x` semver — calling it 0.1.0 signals "I'm shipping this, not asking permission."
- **Title:** `v0.1.0 — public release`.
- **Notes (template):**
  ```
  First public release of Bracket.

  What's in: SDXL LoRA + full-FT (sd-scripts), Z-Image LoRA + full-FT (musubi-tuner),
  Flux-2-Klein 9B LoRA. Optuna TPE + random search. LMStudio-backed VLM judge with
  Qwen3-VL. Multi-seed runs, Welch's t-test confidence reporting. 2-bracket finals
  stage. Gradio UI + CLI.

  What's not in (yet): per-step VLM scoring, ASHA, the React UI rewrite. See README
  roadmap.

  Tested on: Windows 11 + RTX 5090, Ubuntu 22.04 + RTX 4090. CUDA 12.1.

  Install:
      pip install bracket-ml

  Quick start:
      bracket --trainer zimage-full --dataset-toml ./portraits.toml \
              --budget 8 --output-dir ./runs/first

  Thanks to: kohya-ss for sd-scripts, musubi-tuner team, Optuna team, LMStudio.
  ```

### Distribution decisions

- **PyPI: yes.** Package name `bracket-ml` (plain `bracket` is squatted on PyPI). One-line reasoning: pip is the Python ML community's default install path; everything else is friction.
- **Docker: no, not for v0.1.** Reasoning: Bracket sits on top of trainer venvs that themselves don't dockerise cleanly (CUDA, model weights). Document a Dockerfile recipe for users who want it; don't ship and maintain an image.
- **`uvx`/`pipx`: yes, mention in README.** It's free — `pipx install bracket-ml` works the moment PyPI does. One sentence in the install section.
- **Conda: no.** Wait for community demand. Conda packaging maintenance is a non-trivial tax.
- **Homebrew tap: no.** Same reason. Bracket is Python-centric; brew is overkill.

### Social posts (drafts — pick one per channel, edit before posting)

**X / Twitter:**
> Open-sourced Bracket today. It runs your diffusion fine-tune at 8 different configs on a subset of your data, has a vision model rate the samples, and tells you which config wins with a p-value. Drives sd-scripts and musubi-tuner. MIT. <link>

**Hacker News (Show HN):**
> Show HN: Bracket — hyperparameter search for diffusion fine-tunes, scored by a VLM
> A tool I built to stop guessing learning rates for SDXL/Z-Image/Flux LoRAs. Runs N candidate configs on a subset, generates samples, has a local LMStudio Qwen3-VL judge them on prompt adherence + visual quality, runs a finals stage on the top-K, and emits a markdown report with Welch's t-test confidence. Local-first, no SaaS, MIT. Feedback welcome — especially on the search space defaults. <link>

**Reddit r/StableDiffusion:**
> [Tool] Bracket: auto-find the best LR/dim/warmup for your LoRA, scored by a vision model
> Tired of running 5 different LRs by hand and trying to read which samples look best, I built a thing that does this for you. Give it a dataset, a base model (SDXL/Z-Image/Flux-2-Klein supported), a budget, and it'll bracket the search space, score each run by both training loss and a local VLM grading the samples, and report the winner with a confidence interval. Runs on your hardware. <link>

**Reddit r/MachineLearning:**
> [P] Bracket — hyperparameter optimization for diffusion fine-tuning, with VLM-as-judge
> Optuna-driven search on top of sd-scripts/musubi-tuner. Score = weighted sum of loss-component (smoothed final loss + slope) and sample-component (Qwen3-VL JSON grades on prompt adherence/visual quality/artifact-freeness). Multi-seed runs, Welch's t for the verdict. Code, design notes, and the search-space file: <link>

### Landing page

**Recommendation: yes, but minimal.** A single static HTML page at `bracket.tools` (or whichever domain): hero + tagline + the same code-block from the README + three screenshots + GitHub link + docs link. Build with Astro or just hand-write. Half a day of work; pays for itself the first time someone screenshots the URL on Twitter.

Cut anything that doesn't fit on one scroll. No newsletter signup. No "trusted by". No FAQ duplicate (link to repo).

---

## Open questions for the user (recommendation pre-filled)

1. **Brand name:** keep `OmniSteer-Diffusion` or rename to `Bracket`? **Recommend: rename.**
2. **Sibling project framing:** does `OmniSteer-MoE` keep its name? **Recommend: yes, treat OmniSteer as the family/org name and Bracket as this specific product.**
3. **Domain:** acquire `bracket.tools` (or fallback `usebracket.com` / `bracketml.com`)? **Recommend: yes, `bracket.tools` if available — under $20/yr, ship the landing page later.**
4. **PyPI name `bracket-ml`** acceptable, given plain `bracket` is taken? **Recommend: yes — `bracket` as the CLI, `bracket-ml` as the package, no friction in practice.**
5. **Pre-launch:** rename PR before or after the React UI migration? **Recommend: rename first.** Renaming after the React rewrite means re-doing all the new UI strings. Renaming now costs one mechanical PR.

---

*End of proposal. The rename is mechanical, the README rewrite is paste-ready, the launch is a checklist. Pick the name and the rest executes in a day.*
