# Judges — swapping the VLM scorer

Bracket's "judge" scores generated sample images on three axes: prompt adherence, visual quality, and artifact-freeness. The default judge is local LMStudio with a vision model; the protocol is hot-swappable.

## Default: LMStudio + Qwen3-VL

`LMStudioJudge` ([`bracket/judge/lmstudio.py`](../bracket/judge/lmstudio.py)) sends each sample image as a base64-encoded data URL to LMStudio's OpenAI-compatible chat-completions endpoint. The default model is `qwen3-vl-8b-thinking-abliterated` — recommended because it tolerates the wider range of fine-tune content (portraits, NSFW datasets, etc.) without over-refusal.

Other vision models that work out of the box (just change the model name):
- `qwen3-vl-8b` (with-safety variant — refuses on some inputs; flagged in the per-image error)
- `llava-v1.6-mistral-7b`
- `minicpm-v-2.6`

## Configuration

In the UI: Setup tab → "VLM judge (optional)" section. Set `judge_method = lmstudio`, base URL `http://localhost:1234/v1`, model name to whatever LMStudio shows in `/v1/models`.

CLI:
```bash
bracket ... --judge lmstudio --judge-base-url http://localhost:1234/v1 --judge-model qwen3-vl-8b-thinking-abliterated
```

## Memory management

LMStudio holds the model in VRAM after first request. Bracket calls `eject()` on the judge between training runs so the trainer can reclaim that VRAM. Implemented as a POST to `<base_url>/api/v0/models/unload`. Failures are logged and swallowed — the orchestration loop never aborts on a flaky judge control plane.

## Score combination

The combined score is `loss_weight * loss_score + sample_weight * sample_score`. Defaults: `loss_weight=0.3`, `sample_weight=0.7`. Both inputs are normalised to "lower is better" so the combined score is comparable across runs. Override the weights in the UI or via `--loss-weight` / `--sample-weight`.

When the judge isn't configured (or `sample_prompts` wasn't provided), Bracket falls back to `loss_only` scoring. The report says so explicitly — no silent fallback.

## Adding a new judge backend

Implement [`SampleJudge`](../bracket/judge/base.py):

```python
class SampleJudge(ABC):
    @abstractmethod
    def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement: ...
    def eject(self) -> None: ...   # default: no-op
```

`SampleJudgement` carries per-axis scores 0-10 and a free-form `error` field for when something goes wrong on a single image. The orchestrator aggregates judgements per run and rolls them up into the ledger.

Examples worth implementing:
- **OpenAI** (gpt-4o-mini-vision) — fast, cheap, refuses on more content. Set `OPENAI_API_KEY`.
- **Anthropic** (Claude with vision) — strong adherence scoring, also refuses.
- **vLLM** (any open-weight VLM you've quantised yourself) — same OpenAI-compat shape as LMStudio, just point at a different base URL.
- **Local CLIP** — score-from-embedding-distance, very fast, no LLM cost. Different signal — measures alignment, not aesthetic quality.

Register the new judge in `bracket/api/server.py`'s `_start_session_impl()` (the `if judge_method == ...` branch). When the React UI is generated from the OpenAPI schema, the new option will surface automatically in the dropdown.

## Per-image judgement persistence

Every run's `judge_report.judgements` array in the ledger contains `{image, prompt, scores, error, raw_response}` per image. If you want to debug *which* prompt the judge refused on, that's the field to read. The dashboard's "Judge:" status line surfaces aggregate counts; full detail is in the ledger.

## Re-judging samples without re-training

```bash
python scripts/rejudge_run.py <run_dir> <prompts.txt>
```

Reads the run's `output/sample/` images, re-pairs them with the prompts that produced them, sends each through the judge, prints a per-image pass/fail table. Useful when the judge had a transient outage but training succeeded.
