# Roadmap

Bracket is local-first and stays that way. The roadmap below is honest —
only items the maintainer is committed to actually shipping. No
distributed-multi-node, no SaaS, no paid tiers.

## v0.1.0 — first public release (current)

- [x] FastAPI server + WebSocket loss streaming
- [x] React + Vite + shadcn/ui frontend
- [x] Trainer adapters: SDXL LoRA + full FT, Z-Image LoRA + full FT, Flux-2-Klein LoRA
- [x] Optuna TPE search with curated warm-start
- [x] LMStudio VLM judge with hot-swappable backends
- [x] Welch's t-test confidence intervals on multi-seed runs
- [x] Cross-platform installer (Linux / macOS / Windows) with GPU detection

## v0.2 — sharper search

- [ ] **Per-step VLM scoring** — sample at intermediate checkpoints, not just at session end. Catches divergence at step 200 instead of step 500.
- [ ] **True ASHA** (multi-rung promotion, async) — kill bad runs early instead of waiting them out.
- [ ] **Run comparison view in Results** — side-by-side gallery for any 2-3 candidates at the same prompt index.
- [ ] **Frontend deep-link state** — every selection (compared runs, smoothing, theme) reflected in URL.

## v0.3 — more diffusion models

- [ ] **HunyuanDiT** adapter (image, via DiT-Tuner).
- [ ] **Lumina-Next** adapter.
- [ ] **Sana** adapter (efficient text-to-image).
- [ ] **PixArt-Σ** adapter.
- [ ] **AI-Toolkit** as a wrapped trainer — many users use it; Bracket can drive it the same way it drives sd-scripts.

## v0.4 — video diffusion

- [ ] **Wan-2.2** adapter (musubi-tuner already supports it; thin wrapper).
- [ ] **HunyuanVideo** adapter.
- [ ] **CogVideoX** adapter.
- [ ] Video-specific judge prompts (motion coherence, frame consistency, character permanence).

## v0.5 — non-diffusion (LLMs)

This is the biggest leap — different training framework, different scoring axis. Plan:

- [ ] **Axolotl** adapter (most popular OSS LLM fine-tuner).
- [ ] **torchtune** adapter (PyTorch-native, growing fast).
- [ ] **unsloth** adapter (LoRA-only, very VRAM-efficient).
- [ ] **LLM judge protocol** — `LLMJudge` analogous to `SampleJudge`. Scoring axes: perplexity on a held-out set, task-eval suite (GSM8K, MT-Bench), structured-output adherence rate, refusal-rate vs base.
- [ ] **DPO / ORPO / KTO** as training types alongside SFT/LoRA.

## v0.6 — cross-trainer transfer learning

- [ ] **TPE warmup from prior sessions** on related models. Good Z-Image LoRA configs are a useful prior for Flux-2 LoRA. Persist Optuna studies across sessions and replay matching trials when a new session starts.

## Not on the roadmap

- Distributed multi-node training. Bracket runs trials sequentially on one box; that's the design.
- Cloud bursting. Local-first.
- Paid tier / hosted version. The maintainer isn't building a SaaS.
- Replacing the trainers. Bracket drives sd-scripts / musubi-tuner / etc; it doesn't compete with them.

## How to influence this list

- **Existing trainer not supported?** Open a PR adding the adapter. See [`.claude/skills/bracket-add-trainer/SKILL.md`](../.claude/skills/bracket-add-trainer/SKILL.md).
- **New search algorithm?** PR a new `SearchController` implementation in `bracket/search/`.
- **Different VLM?** Implement `SampleJudge` in `bracket/judge/`. Hot-swappable.
- **Anything else**: open a discussion before a PR for items that span subsystems.
