"""No-reference image quality judge built on `pyiqa`.

CLIP-IQA (or sibling metrics MUSIQ / MANIQA, also exposed by pyiqa) is
NOT used as a primary ranker — it carries known biases (favours saturated
high-contrast images, dislikes intentional film grain, doesn't really
"understand" the prompt). The point is to catch cases where the VLM
hallucinates that a melted/garbled image is fine.

This judge implements the SampleJudge protocol so it can run alongside
the LMStudio judge and emit a `clip_iqa_median` component on the
ScoreReport. The orchestrator may also use it as a disqualification
gate: when the median CLIP-IQA score across an image set drops below
``clip_iqa_dq_threshold`` (default 0.30), the run is tagged with
``disqualified = "low_clip_iqa"``.

The pyiqa import is lazy and lives inside ``_load_model`` so this module
can be imported (and unit-tested in places that don't have pyiqa
installed) without paying the torch import cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bracket.judge.base import SampleJudge, SampleJudgement

logger = logging.getLogger("bracket.judge.clip_iqa")


@dataclass(frozen=True)
class ClipIqaJudgeConfig:
    """Configuration for the CLIP-IQA / pyiqa-backed judge.

    ``metric`` is forwarded directly to ``pyiqa.create_metric``. Default
    ``clipiqa+`` is the multi-aspect variant CLIP-IQA paper recommends
    (sharper signal on aesthetic vs. structural artefacts than plain
    ``clipiqa``). Other reasonable choices for our DQ use-case:
    ``musiq``, ``maniqa``, ``brisque``.
    """

    metric: str = "clipiqa+"
    device: str = "cuda"
    # When the median CLIP-IQA score across the run's sample set falls
    # below this, the orchestrator's gate path tags the run with
    # ``disqualified = "low_clip_iqa"``. Scale: pyiqa returns values in
    # [0, 1] for clipiqa+ (higher = better quality).
    dq_threshold: float = 0.30


class ClipIqaJudge(SampleJudge):
    """No-reference quality judge using pyiqa's CLIP-IQA metric.

    Each call to ``judge_image`` returns a SampleJudgement where the
    ``visual_quality`` axis carries the raw CLIP-IQA score (clipped to
    [0, 10] by mapping the [0, 1] metric output through ×10). The other
    axes are repeated for compatibility with the JudgeReport aggregation
    code — CLIP-IQA doesn't know about prompts so prompt_adherence is
    set equal to the quality score.
    """

    def __init__(self, config: Optional[ClipIqaJudgeConfig] = None) -> None:
        self.config = config or ClipIqaJudgeConfig()
        self._metric = None
        self._import_error: Optional[Exception] = None

    def _load_model(self) -> None:
        if self._metric is not None or self._import_error is not None:
            return
        try:
            import pyiqa  # type: ignore[import-not-found]
        except ImportError as e:  # noqa: BLE001 — re-raise as helpful message later
            self._import_error = ImportError(
                "pyiqa is not installed. Install Bracket with the "
                "'vision_metrics' extra (pip install bracket-ml[vision_metrics]) "
                "or `pip install pyiqa` directly."
            )
            return
        try:
            self._metric = pyiqa.create_metric(
                self.config.metric, device=self.config.device,
            )
        except Exception as e:  # noqa: BLE001 — bubble up as runtime error per-image
            self._import_error = e

    def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
        image_path = Path(image_path)
        self._load_model()
        if self._metric is None:
            err = self._import_error or RuntimeError("clip-iqa metric not initialised")
            logger.warning(
                "CLIP-IQA judge unavailable for %s: %s", image_path.name, err,
            )
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=0.0, visual_quality=0.0,
                artifact_free=0.0, overall=0.0,
                raw_response="",
                error=f"{type(err).__name__}: {err}",
            )
        try:
            score_01 = float(self._metric(str(image_path)).item())
        except Exception as e:  # noqa: BLE001 — pyiqa / torch can raise broadly
            logger.warning("CLIP-IQA failed on %s: %s", image_path.name, e)
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=0.0, visual_quality=0.0,
                artifact_free=0.0, overall=0.0,
                raw_response="",
                error=f"{type(e).__name__}: {e}",
            )
        score_01 = max(0.0, min(1.0, score_01))
        score_10 = score_01 * 10.0
        return SampleJudgement(
            image_path=image_path, prompt=prompt,
            prompt_adherence=score_10,
            visual_quality=score_10,
            artifact_free=score_10,
            overall=score_10,
            raw_response=f"clip_iqa_raw={score_01:.4f}",
        )

    def eject(self) -> None:
        """Free the pyiqa model + torch CUDA cache between training runs."""
        if self._metric is None:
            return
        try:
            del self._metric
        finally:
            self._metric = None
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001 — eject must never raise
                logger.debug("torch.cuda.empty_cache() failed (ignored): %s", e)


def median(values: list[float]) -> float:
    """Plain-Python median — avoids dragging in statistics for one call."""
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0
