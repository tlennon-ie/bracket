"""Tests for ClipIqaJudge — the no-reference image-quality DQ gate.

Real pyiqa lives behind an optional `vision_metrics` extra and pulls
torch + open-clip in. The unit suite must stay fast and dependency-free,
so we exercise the judge by monkey-patching its lazy metric loader to
return a stub callable. End-to-end behaviour (gate emits the median
component, DQs below threshold) is verified through this stub.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bracket.judge.base import SampleJudge, SampleJudgement
from bracket.judge.clip_iqa import ClipIqaJudge, ClipIqaJudgeConfig, median


def _write_image(p: Path) -> None:
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes-for-testing")


class _StubMetric:
    """Drop-in for ``pyiqa.create_metric(...)`` return: callable that
    yields a tensor-like object exposing ``.item()``."""

    def __init__(self, score_per_path: dict[str, float]) -> None:
        self._scores = score_per_path

    def __call__(self, path: str) -> "_StubMetric":  # type: ignore[override]
        # Return self so the next ``.item()`` reads the matching value.
        self._current = float(self._scores.get(path, self._scores.get("*", 0.5)))
        return self

    def item(self) -> float:
        return self._current


def test_clip_iqa_judge_returns_float_in_0_to_1_range_after_normalisation(
    tmp_path: Path, monkeypatch,
):
    """End-to-end stub test: judge_image must return a SampleJudgement
    whose overall (= raw 0..1 score × 10) is a finite float in [0, 10]."""
    img = tmp_path / "a.png"; _write_image(img)
    judge = ClipIqaJudge(ClipIqaJudgeConfig(metric="clipiqa+", device="cpu"))
    monkeypatch.setattr(judge, "_load_model", lambda: None)
    judge._metric = _StubMetric({str(img): 0.72})  # type: ignore[assignment]

    res = judge.judge_image(img, "a beach scene")
    assert res.error is None
    assert isinstance(res.overall, float)
    assert 0.0 <= res.overall <= 10.0
    # Raw value passes through visual_quality as well so the JudgeReport
    # aggregator can mean-it for display.
    assert res.visual_quality == pytest.approx(7.2, rel=1e-6)
    assert "clip_iqa_raw" in res.raw_response


def test_clip_iqa_judge_reports_import_error_helpfully(tmp_path: Path, monkeypatch):
    """When pyiqa isn't installed, judge_image must return an errored
    judgement (not raise) so the orchestrator keeps moving."""
    img = tmp_path / "a.png"; _write_image(img)
    judge = ClipIqaJudge()

    # Simulate the import failure path.
    monkeypatch.setattr(judge, "_load_model", lambda: None)
    import bracket.judge.clip_iqa as ci
    judge._import_error = ImportError(  # type: ignore[assignment]
        "pyiqa not installed",
    )

    res = judge.judge_image(img, "any")
    assert res.error is not None
    assert "pyiqa" in res.error
    assert res.overall == 0.0


def test_clip_iqa_judge_clips_extreme_metric_values(tmp_path: Path, monkeypatch):
    """pyiqa could return values slightly outside [0, 1] depending on
    the metric; clip them so we never overflow the 0..10 axis."""
    img = tmp_path / "a.png"; _write_image(img)
    judge = ClipIqaJudge()
    monkeypatch.setattr(judge, "_load_model", lambda: None)
    judge._metric = _StubMetric({str(img): 1.5})  # type: ignore[assignment]

    res = judge.judge_image(img, "any")
    assert res.overall == pytest.approx(10.0)


def test_scorer_clip_iqa_gate_dqs_below_threshold(tmp_path: Path):
    """When the secondary CLIP-IQA judge returns a median below the
    configured DQ threshold, the run must be tagged disqualified =
    'low_clip_iqa' and score=+inf."""
    from bracket.orchestrator.scorer import Scorer
    import time as _t

    # Build a tfevents file converging to a non-disqualified loss.
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary

    ev_dir = tmp_path / "ev"
    ev_dir.mkdir()
    writer = EventFileWriter(str(ev_dir), max_queue_size=512, flush_secs=0)
    base = _t.time()
    for step in range(1, 60):
        writer.add_event(Event(
            wall_time=base + step * 0.01, step=step,
            summary=Summary(value=[Summary.Value(
                tag="loss/current", simple_value=0.4 - 0.001 * step,
            )]),
        ))
    writer.close()
    tfe = next(p for p in ev_dir.glob("events.out.tfevents.*"))

    # Sample dir with paired images.
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    for idx in range(3):
        (sample_dir / f"candidate_000040_{idx}_42.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    # Primary VLM judge: gives the run great scores.
    class _GoodVLM(SampleJudge):
        def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=8.0, visual_quality=8.0,
                artifact_free=8.0, overall=8.0, raw_response="",
            )

    # CLIP-IQA secondary: reports a median below threshold so the
    # gate must kick in despite the VLM saying everything's fine.
    class _LowQualityClipIqa(SampleJudge):
        def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
            # 0.10 raw → 1.0 on the 0..10 axis. Well below threshold.
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=1.0, visual_quality=1.0,
                artifact_free=1.0, overall=1.0, raw_response="clip_iqa_raw=0.1000",
            )

    scorer = Scorer(
        sample_judge=_GoodVLM(),
        loss_weight=0.3, sample_weight=0.7,
        clip_iqa_judge=_LowQualityClipIqa(),
        clip_iqa_dq_threshold=0.3,
    )
    rep = scorer.score_run(
        tfevents_path=tfe, sample_dir=sample_dir,
        prompts=["a beach", "a forest", "a city"],
    )
    assert rep.disqualified == "low_clip_iqa"
    assert rep.score == float("inf")
    assert "clip_iqa_median" in rep.components
    assert rep.components["clip_iqa_median"] < 0.3


def test_scorer_clip_iqa_gate_passes_above_threshold(tmp_path: Path):
    """When CLIP-IQA reports high quality, the run is NOT disqualified;
    the median is still emitted as a component for transparency."""
    from bracket.orchestrator.scorer import Scorer
    import time as _t
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary

    ev_dir = tmp_path / "ev"
    ev_dir.mkdir()
    writer = EventFileWriter(str(ev_dir), max_queue_size=512, flush_secs=0)
    base = _t.time()
    for step in range(1, 60):
        writer.add_event(Event(
            wall_time=base + step * 0.01, step=step,
            summary=Summary(value=[Summary.Value(
                tag="loss/current", simple_value=0.4 - 0.001 * step,
            )]),
        ))
    writer.close()
    tfe = next(p for p in ev_dir.glob("events.out.tfevents.*"))

    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    (sample_dir / "candidate_000040_0_42.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")

    class _GoodVLM(SampleJudge):
        def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=8.0, visual_quality=8.0,
                artifact_free=8.0, overall=8.0, raw_response="",
            )

    class _GoodClipIqa(SampleJudge):
        def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=7.5, visual_quality=7.5,
                artifact_free=7.5, overall=7.5, raw_response="clip_iqa_raw=0.7500",
            )

    scorer = Scorer(
        sample_judge=_GoodVLM(),
        clip_iqa_judge=_GoodClipIqa(),
        clip_iqa_dq_threshold=0.3,
    )
    rep = scorer.score_run(
        tfevents_path=tfe, sample_dir=sample_dir, prompts=["a beach"],
    )
    assert rep.disqualified is None
    assert rep.components["clip_iqa_median"] == pytest.approx(0.75)


def test_median_helper():
    assert median([]) == 0.0
    assert median([5.0]) == 5.0
    assert median([1.0, 2.0, 3.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
