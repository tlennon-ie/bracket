# Orchestrator design

## What it is

A loop over: **pick → launch → score → record → repeat**. Each iteration is
one real training run on a small dataset subset. The agent's job is to find
a config that beats the hand-tuned baseline under a fixed budget.

## What problem it solves

Diffusion fine-tuning has too many knobs (learning rate, optimizer, scheduler,
warmup, rank, alpha, noise offset, weighting scheme, mixed precision …) and
no closed-form way to know which combination is right for *your* dataset and
*your* base model. Today that's a multi-day human exercise of "tweak, train,
look at samples, tweak again." The orchestrator does that loop autonomously.

## Why each component looks the way it does

### Trainer protocol (instead of hard-coding sd-scripts)

Diffusion fine-tuning trainers all look similar from the outside —
`accelerate launch <script> --pretrained_model … --dataset_config … --max_train_steps …` —
but their argparser surface and TOML schemas differ in detail. The Trainer
abstraction lets the orchestrator drive **any** of them with a thin
per-trainer adapter:

```python
class Trainer(ABC):
    def declare_search_space(self) -> SearchSpace: ...
    def baseline_config(self) -> TrainerConfig: ...
    def config_from_dict(self, knobs) -> TrainerConfig: ...
    def prepare_run(self, *, run_dir, config, ...) -> LaunchSpec: ...
```

`SDXLTrainer` is the v0.2 implementation. v0.5 adds `Flux2KleinTrainer`
against musubi-tuner's `flux_2_train_network.py` with no changes elsewhere.

### Search space lives on the trainer (not the orchestrator)

Different base models have different sweet spots. SDXL LoRA wants
lr ∈ [1e-6, 5e-4] with AdamW8bit/Lion/Prodigy; Flux 2 fp8 wants lr ∈ [1e-7, 5e-5]
with mostly Adafactor. So `declare_search_space()` is a method on the trainer,
and the user's "perfect config per dataset per model" becomes well-defined:
the orchestrator searches within whatever range the trainer claims is sane.

### Score = final smoothed loss + |slope of last 25%|

We need a *cheap* signal — running a VLM judge on every sample image of every
candidate would multiply wall time. EMA-smoothed final loss captures "where
it ended up." The slope term punishes runs that ended low only because they
hadn't started diverging yet. Together: lower final + flat slope = good.

Disqualification rules (score = +∞):
- No tfevents written → trainer crashed before any logging
- Empty tfevents → trainer crashed during init  
- NaN final loss → numerical failure
- Positive slope > kill threshold → optimizer blew up

This is intentionally conservative — we'd rather throw out a borderline
divergent run than declare it a winner. The kill threshold can be tuned.

### Random search first (NOT bandit/Bayesian)

In low-data regimes (≤20 candidates), random search is competitive with
fancier methods *and* it's transparent. v0.1 ships with random because:
- Zero hyperparameters of its own (no acquisition function to tune)
- Easy to reproduce (just a seed)
- Establishes the ledger format that future controllers consume

The `SearchController` interface admits future controllers without touching
the loop:

```python
class SearchController(ABC):
    def next_config(self, space, history) -> dict: ...
    def should_stop(self, history, *, budget_runs) -> bool: ...
```

A bandit controller in v0.4 reads the ledger's history (config → score) and
biases sampling toward promising regions; the loop doesn't change.

### Ledger is JSONL, not SQLite

Why: 
- One row per run, append-only, crash-safe.
- Easy to grep / pipe into `jq`.
- Resumable via re-reading: if the orchestrator dies, the next invocation
  reads existing rows and picks up after the last successful candidate.
- No schema migration burden.
- A SaaS-tier database can ingest these JSONL files later — no rewrite.

Floats that JSON can't represent (`inf`, `nan`) are stringified on write and
treated as "no score" on read. Partial trailing lines (write interrupted by
crash) are tolerated.

### Subprocess management is its own module

Because Windows. Per-run we need to:
- Capture stdout to a per-run log file (no shared output buffer).
- Stream stdout in a thread so the OS pipe buffer doesn't fill and deadlock.
- Kill the entire process tree on timeout, including accelerate's worker
  threads. On Windows that's `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT`;
  on POSIX it's `os.setsid` + `os.killpg`.
- Tolerate UTF-8 / cp1252 encoding mismatches in trainer output.

`RunLauncher` encapsulates all of this. Tests use Python itself as a fake
trainer to validate the lifecycle without needing a GPU.

### Dataset subset is materialized to disk

Sd-scripts (and musubi-tuner) cache latents and TE outputs adjacent to the
images. If two candidates point at *the same* image directory, sd-scripts
reuses the cache. If two candidates use different images, the cache rebuilds.
We materialize **one shared subset** for the whole orchestration session, so
the cache is built once on the first run and reused by every candidate.

This is also why `cache_directory` in the source TOML is *not* honored — the
cache is per-session, not per-source-dataset.

### Proof report is Markdown (not a dashboard)

Two reasons:
1. The proof artifact has to survive the session — it's the "did orchestrator
   beat baseline" answer the user reads tomorrow. Markdown is grep-able,
   diff-able, embeds in PRs.
2. Live monitoring is the v0.1 telemetry dashboard's job (see `frontend/`).
   The report is post-hoc.

## What's deliberately not in v0.2

- **VLM judge.** Loss is cheap and bounded; VLM scores cost API calls or
  local GPU time, scale poorly to 20+ candidates, and need careful prompt
  design. v0.3.
- **Multi-trainer.** Sd-scripts SDXL only. Flux 2 Klein adapter is v0.5.
- **Parallel candidates.** Sequential keeps the GPU memory model trivial.
- **Live in-process steering of an active run.** Out of scope; the unit of
  control is the run, not the step.
- **Bayesian / bandit / LLM-guided search.** Random search is the v0.1 baseline
  to compare future controllers against.
- **Multi-tenant / SaaS.** Phase 0.7+; the ledger format is forward-compatible.

## Extension points

Each labeled module is a thin contract you can replace:

| Replace | Interface | Example future work |
|---|---|---|
| Trainer | `Trainer` ABC | Flux 2 Klein, Z-Image LoRA, SD3, kohya OneTrainer |
| Search controller | `SearchController` ABC | Bandit, BO, LLM-guided |
| Scorer | `Scorer.score_tfevents()` | Add VLM-judged sample quality term |
| Ledger | `Ledger.append/iter_rows` | SQLite, Postgres, Vercel KV |
| Proof | `proof.generate_report()` | Plotly HTML, PDF, Slack post |

The loop in `orchestrator/loop.py` doesn't know about any of these specifics
— it just composes them.
