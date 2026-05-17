"""SampleJudge abstraction — a vision-language model that scores generated
images against the prompts that produced them.

Why an abstraction: today this is LMStudio + Qwen3-VL. Tomorrow the user may
want a hosted call (Anthropic, OpenAI), a different local model (vLLM), or
a no-op stub for testing. The orchestrator only depends on the protocol.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Protocol


@dataclass(frozen=True)
class SampleJudgement:
    """Per-axis scores for one generated image. All axes 0..10, higher better."""

    image_path: Path
    prompt: str
    prompt_adherence: float   # how well the image matches the prompt
    visual_quality: float     # sharpness, composition, anatomy
    artifact_free: float      # absence of AI artifacts (warped faces, garbled text)
    overall: float            # weighted combination — judge-defined
    raw_response: str = ""    # the model's raw JSON / text output, for debugging
    error: Optional[str] = None  # set if parsing/scoring failed; overall=0 in that case
    # Self-consistency uncertainty: max std across the four scored axes
    # when the judge was sampled N>1 times at a non-zero temperature.
    # 0.0 means a single deterministic sample (the default) — same value
    # the old field-less code would have implied.
    score_variance: float = 0.0
    # True when the per-image variance crossed the configured threshold;
    # surfaced in the proof report so users can see which images the
    # VLM disagreed with itself on.
    judge_uncertain: bool = False


@dataclass(frozen=True)
class JudgeReport:
    """Aggregate of N per-image judgements for one training run."""

    n_images: int
    n_failed: int            # judgements that errored out (parse failure, API down)
    mean_overall: float      # 0..10
    mean_prompt_adherence: float
    mean_visual_quality: float
    mean_artifact_free: float
    judgements: tuple[SampleJudgement, ...] = field(default_factory=tuple)

    @property
    def normalized_score(self) -> float:
        """Map mean_overall (0..10) to a quantity where lower is better, so the
        orchestrator's combined score (loss-based component is also lower-better)
        composes naturally."""
        return 1.0 - (self.mean_overall / 10.0)


class SampleJudge(ABC):
    """Score one image given the prompt that generated it.

    Implementations should be reasonably robust: a transient API error on one
    image must not abort the whole run's scoring — return a SampleJudgement
    with `error` set and `overall=0` so the aggregate at least includes the
    other images.
    """

    @abstractmethod
    def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
        """Score a single image. Synchronous so the scorer doesn't pull in
        async machinery for what's already a slow IO-bound call."""

    def eject(self) -> None:
        """Release whatever VRAM/process resources the judge is holding.

        Default no-op so simple judges don't have to care. Local-model judges
        (LMStudio, vLLM) override this to free GPU memory before the next
        training run starts. Must be safe to call multiple times and never
        raise — failures should be logged and swallowed.
        """
        return None

    def judge_run(self, sample_dir: Path, prompt_for_image: dict[Path, str]) -> JudgeReport:
        """Score every image in `sample_dir` whose path appears in
        `prompt_for_image`. Returns the aggregate."""
        judgements: list[SampleJudgement] = []
        n_failed = 0
        for img_path, prompt in prompt_for_image.items():
            if not img_path.exists():
                continue
            try:
                j = self.judge_image(img_path, prompt)
            except Exception as e:  # noqa: BLE001 — judges are external and flaky
                j = SampleJudgement(
                    image_path=img_path, prompt=prompt,
                    prompt_adherence=0.0, visual_quality=0.0, artifact_free=0.0,
                    overall=0.0, raw_response="", error=f"{type(e).__name__}: {e}",
                )
            if j.error:
                n_failed += 1
            judgements.append(j)
        if not judgements:
            return JudgeReport(
                n_images=0, n_failed=0,
                mean_overall=0.0, mean_prompt_adherence=0.0,
                mean_visual_quality=0.0, mean_artifact_free=0.0,
                judgements=(),
            )
        scored = [j for j in judgements if j.error is None]
        if not scored:
            return JudgeReport(
                n_images=len(judgements), n_failed=n_failed,
                mean_overall=0.0, mean_prompt_adherence=0.0,
                mean_visual_quality=0.0, mean_artifact_free=0.0,
                judgements=tuple(judgements),
            )
        return JudgeReport(
            n_images=len(judgements),
            n_failed=n_failed,
            mean_overall=sum(j.overall for j in scored) / len(scored),
            mean_prompt_adherence=sum(j.prompt_adherence for j in scored) / len(scored),
            mean_visual_quality=sum(j.visual_quality for j in scored) / len(scored),
            mean_artifact_free=sum(j.artifact_free for j in scored) / len(scored),
            judgements=tuple(judgements),
        )


# ─── Pairwise judging ─────────────────────────────────────────────


# Result of a single pairwise A/B comparison.
#   -1 = image A wins, 0 = tie, +1 = image B wins.
PairwiseOutcome = Literal[-1, 0, 1]


class PairwiseJudge(Protocol):
    """A judge that prefers one image over another given a shared prompt.

    Implementations should be best-effort — a transient API error must
    not abort an entire tournament. On failure the implementation may
    return 0 (tie) and log the error; callers should treat ties as
    "no information" when fitting the Bradley-Terry model.
    """

    def judge_pair(
        self, image_a: Path, image_b: Path, prompt: str,
    ) -> PairwiseOutcome: ...

    def eject(self) -> None:
        """Release resources (e.g. VLM VRAM). Default no-op for stubs."""


# Sample-prompts files in sd-scripts / musubi-tuner format are one prompt per
# line, with optional `--w 1024 --h 1024 --s 20 --d 42` etc. flags. We strip
# those flags and # comments to get clean prompt strings for the VLM.
_FLAG_RE = re.compile(r"\s+--\w+\s+\S+")


def parse_judge_prompts_file(path: Path) -> list[str]:
    """Read sd-scripts-style sample_prompts.txt → list of clean prompt strings."""
    out: list[str] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        clean = _FLAG_RE.sub("", line).strip()
        if clean:
            out.append(clean)
    return out
