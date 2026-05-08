# UI guide

Launch:

```powershell
& "I:/AI/musubi-tuner/venv/Scripts/python.exe" -m bracket.ui.app
# http://127.0.0.1:7860
```

## Tab 1 — Setup

The setup tab captures everything that doesn't change between sessions:
where your tools live, where your models live, where your dataset is.

Required for **all trainers**:
- **venv python.exe** — usually `I:/AI/musubi-tuner/venv/Scripts/python.exe`
- **Dataset TOML** — your existing musubi or sd-scripts dataset config
- **Sample prompts** — needed if you want VLM judging or visual comparison
- **Session output dir** — where everything for this run lands

For **sdxl-lora / sdxl-full**:
- **sd-scripts dir** — usually `I:/AI/musubi-tuner/sd-scripts`
- **SDXL base** — HF snapshot directory (or single safetensors)

For **zimage-lora / zimage-full**:
- **musubi-tuner dir** — `I:/AI/musubi-tuner`
- **DiT weights** — e.g. `z_image_bf16.safetensors`
- **VAE weights** — `ae.safetensors`
- **Text encoder** — `qwen_3_4b.safetensors` or `qwen_3_8b_fp8mixed.safetensors`

VRAM override defaults to 0 (auto-detect from CUDA). Set explicitly only if
you want to keep margin for other GPU users (e.g. enter 24 on a 32 GB card).

VLM judge: leave at "none" for fast loss-only scoring; switch to "lmstudio"
to enable Qwen3-VL judging (start LMStudio first with the model loaded).
Loss/sample weight controls how much the two components blend — defaults
0.3 / 0.7 favor sample quality.

## Tab 2 — Run

- **Candidates** — number of unique configs to try. 4-8 for a quick test;
  16-32 for a real session.
- **Seeds per config** — 1 = fast, no confidence intervals. 2-3 = enables
  Welch's t-test verdicts. Triples wall time.
- **Max steps per run** — 200-400 for cheap ranking; 800+ if you don't trust
  short-run signal.
- **Search method** — "optuna" (smart) or "random" (baseline).
- **Finals top-K** — set to 0 to skip the finals stage. Set to 3 to take the
  top 3 stage-1 winners and re-run them at `Finals max steps` for verification.

Press **Start orchestration** — output goes to the Monitor and Results tabs.

## Tab 3 — Monitor

Auto-refreshes every 2 seconds.

- **Status line** — current phase (running / done / error), runs completed,
  elapsed wall time, last meaningful log line
- **Log tail** — last ~400 lines from the orchestrator and trainer
  subprocesses

If status hits `error`, the full traceback is shown inline.

## Tab 4 — Results

- **Report** — the auto-generated `report.md`. Headline (orchestrator vs
  baseline), confidence verdict (when seeds ≥ 2), top-5, sample quality
  breakdown when judge ran, disqualifications, score histogram.
- **Ledger** — every row of `ledger.jsonl` as a sortable table.
- **Samples gallery** — final-state samples from each run, browsable.
  Hover to see which run produced each.

Press **Refresh results** after the session finishes.

## Resuming after a crash / Ctrl-C

The ledger is append-only. If you stop a session and re-run with the same
output dir, Bracket picks up where it left off. The baseline
isn't re-run; missing candidates are filled in. The report regenerates from
the merged ledger.

## What good looks like in the report

A successful session ends with something like:

```
**Orchestrator beat baseline.**

| | run_id | mean | stdev | n_seeds |
|---|---|---|---|---|
| baseline | baseline-000-...    | 0.341... | 0.004 | 2 |
| **best** | cand-002-...        | 0.241... | 0.011 | 2 |

Δ score: -0.100 (-29.3%)

**Confidence (best vs baseline):**
- Welch's t-test p-value: 0.003
- 95% CI for Δ score: [-0.142, -0.058]
- Verdict: **high confidence** (p<0.01)
```

Translation: "the config the agent picked beat your hand-tuned baseline by
~29% on our combined score, and we're statistically confident in that gap
(p<0.01) — not a fluke of one good seed."

If instead the verdict reads `noisy — extend budget / seeds to confirm`,
re-run with more seeds-per-config (the resume will only re-run candidates;
to re-seed, change `--output-dir`).
