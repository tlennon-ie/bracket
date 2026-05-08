---
name: bracket-debug-run
description: Use when a Bracket run is failing, hanging, producing no scores, or the dashboard is showing weird state. Walks through ledger inspection, log triage, and the most common failure classes.
---

# Bracket — debugging a failing or stuck run

The two artefacts that matter for debugging:
1. **`<output-dir>/ledger.jsonl`** — one JSON line per run.
2. **`<output-dir>/runs/<run_id>/logs/stdout.log`** — the trainer subprocess's full stdout.

Always start by reading the ledger.

## Read the ledger

```bash
tail -n 20 <output-dir>/ledger.jsonl | python -m json.tool --json-lines
```

Each row tells you: `role` (setup / baseline / curated / candidate / finalist), `score` (or null if disqualified), `score_components` (loss + judge breakdown), `judge_report` (or null), `disqualified` reason, `exit_code`, `duration_s`.

## Common patterns

### Pattern: every candidate has `score = null`, `disqualified = "empty_tfevents"`

The trainer subprocess is failing instantly. Read its log:
```bash
tail -n 100 <output-dir>/runs/<run_id>/logs/stdout.log
```

Most common causes (by frequency):
- **`KeyError: 'beta1'`** with `--fused_backward_pass` — AdamW + fused is broken in musubi's zimage_train.py. Bracket already excludes AdamW from Z-Image full-FT search space; if you see this, you're on an outdated build or a custom trainer.
- **`FileNotFoundError`** on a `.safetensors` path — the user's `BRACKET_*_PATH` env var or the field on the Setup tab is wrong.
- **CUDA OOM** — batch size too high. Bracket's search space is VRAM-tier-aware but a candidate may still OOM. Reduce `--seeds-per-config`, drop `--budget`, or set a smaller VRAM tier in the trainer config.

### Pattern: all candidates score, but `judge_report` is null on every row

The VLM judge wasn't called. Causes:
- `judge_method != "lmstudio"` (CLI default) or judge field empty (UI).
- `sample_prompts` not provided — without it, Bracket falls back to loss-only scoring.
- LMStudio not running on the configured port. Test: `curl http://localhost:1234/v1/models`.
- Sample-image filename pattern unrecognised by `_pair_samples_with_prompts` in [`bracket/orchestrator/scorer.py`](../../bracket/orchestrator/scorer.py). This shouldn't happen for sd-scripts or musubi outputs but might if a custom trainer writes images differently.

### Pattern: dashboard stuck at e.g. "5/9" while training is finishing the 9th run

This was a real bug in pre-rebrand versions where curated runs weren't counted in `completed_runs`. Fixed at [`bracket/ui/monitor.py`](../../bracket/ui/monitor.py) (constant `_TRAINING_ROLES`). If you see this on a current build, check that role.

### Pattern: 2000-4000 s/it instead of 2-3 s/it on Blackwell

This is the "stdout pipe deadlock" issue. The trainer must write directly to a log file, not through `subprocess.PIPE`. Check that `RunLauncher` in [`bracket/orchestrator/runner.py`](../../bracket/orchestrator/runner.py) opens the log file with `subprocess.Popen(stdout=open(log_path, "ab"))`, NOT `stdout=subprocess.PIPE`. The fix is documented in its docstring.

### Pattern: cand-001 / cand-004 / etc. dying after ~30s with exit_code=1

Usually a config the trainer literally can't accept. The Adafactor + fused_backward_pass + AdamW combo is the canonical example. Look at the disqualified row's `config` — what knob value is unusual? Cross-reference with the trainer's documented arg surface.

## Re-judging old samples

If the judge had a transient outage but training succeeded, you can re-score samples without re-training:

```bash
python scripts/rejudge_run.py <run_dir> <prompts.txt>
```

This reads the run's `output/sample/` images, sends each to LMStudio paired with its prompt, and prints a per-image pass/fail table. Use it to confirm the judge is healthy without spending another budget.

## When in doubt

Open the FastAPI server and curl the live snapshot:

```bash
curl http://127.0.0.1:8000/api/session | python -m json.tool
curl http://127.0.0.1:8000/api/judge/status | python -m json.tool
curl http://127.0.0.1:8000/api/runs | python -m json.tool
```

These return the exact data the React UI is displaying. If they look right but the UI looks wrong, it's a frontend bug; if they look wrong, it's a backend bug.
