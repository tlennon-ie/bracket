"""Scorer tests with VLM judge — ensures sample quality and loss combine
correctly, and that we fall back gracefully when samples or judge are absent."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from bracket.judge.base import SampleJudge, SampleJudgement
from bracket.orchestrator.scorer import Scorer, _pair_samples_with_prompts


class _FixedJudge(SampleJudge):
    def __init__(self, score: float = 8.0) -> None:
        self.score = score

    def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
        return SampleJudgement(
            image_path=image_path, prompt=prompt,
            prompt_adherence=self.score, visual_quality=self.score,
            artifact_free=self.score, overall=self.score,
            raw_response="",
        )


def _write_tfevents(tmp: Path, losses: list[float]) -> Path:
    writer = EventFileWriter(str(tmp), max_queue_size=512, flush_secs=0)
    base = time.time()
    for step, loss in enumerate(losses, start=1):
        writer.add_event(Event(
            wall_time=base + step * 0.01, step=step,
            summary=Summary(value=[Summary.Value(tag="loss/current", simple_value=loss)]),
        ))
    writer.close()
    return next(p for p in tmp.glob("events.out.tfevents.*"))


def _sample_at(dir_: Path, name: str) -> Path:
    p = dir_ / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 16)
    return p


def test_scorer_falls_back_to_loss_only_when_no_judge(tmp_path: Path):
    tfe = _write_tfevents(tmp_path / "ev", [0.4 - 0.001 * i for i in range(50)])
    scorer = Scorer()
    rep = scorer.score_run(tfevents_path=tfe, sample_dir=None, prompts=None)
    assert rep.disqualified is None
    assert "loss_score" not in rep.components
    assert "loss_only" in rep.components


def test_scorer_combines_loss_and_sample(tmp_path: Path):
    tfe = _write_tfevents(tmp_path / "ev", [0.4 - 0.001 * i for i in range(50)])
    sample_dir = tmp_path / "samples"; sample_dir.mkdir()
    # sd-scripts naming: <output_name>_<step>_<idx>_<seed>.png
    _sample_at(sample_dir, "candidate_000040_0_42.png")
    _sample_at(sample_dir, "candidate_000040_1_42.png")
    judge = _FixedJudge(score=8.0)
    scorer = Scorer(sample_judge=judge, loss_weight=0.3, sample_weight=0.7)
    rep = scorer.score_run(
        tfevents_path=tfe, sample_dir=sample_dir,
        prompts=["a beach", "a portrait"],
    )
    assert rep.disqualified is None
    assert rep.judge_report is not None
    assert rep.judge_report.n_images == 2
    # sample component: 1 - 8/10 = 0.2; loss is ~0.35; combined weighted-mean
    assert "loss_score" in rep.components
    assert "sample_score" in rep.components
    # combined < loss alone (sample quality is good, dragging score down)
    assert rep.score < rep.components["loss_score"]


def test_scorer_handles_empty_sample_dir(tmp_path: Path):
    tfe = _write_tfevents(tmp_path / "ev", [0.4 - 0.001 * i for i in range(50)])
    sample_dir = tmp_path / "samples"; sample_dir.mkdir()
    judge = _FixedJudge()
    scorer = Scorer(sample_judge=judge)
    rep = scorer.score_run(
        tfevents_path=tfe, sample_dir=sample_dir,
        prompts=["p1", "p2"],
    )
    assert "sample_dir_empty" in rep.components
    assert rep.judge_report is None


def test_pair_samples_picks_correct_prompt_by_index(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    _sample_at(sd, "candidate_000200_0_42.png")
    _sample_at(sd, "candidate_000200_1_42.png")
    _sample_at(sd, "candidate_000200_2_42.png")
    pairing = _pair_samples_with_prompts(sd, ["A", "B", "C"])
    assert len(pairing) == 3
    assert any(v == "A" for v in pairing.values())
    assert any(v == "C" for v in pairing.values())


def test_pair_samples_handles_musubi_filename_format(tmp_path: Path):
    """musubi writes <name>_<step>_<idx>_<YYYYMMDDhhmmss>_<seed>_<batch>.png.

    Regression: prior pairing logic walked from the right and treated the
    seed as the prompt index, so every musubi sample dropped out and the
    judge silently never got called.
    """
    sd = tmp_path / "samples"; sd.mkdir()
    _sample_at(sd, "candidate_000300_00_20260508122543_42_000.png")
    _sample_at(sd, "candidate_000300_01_20260508122600_43_000.png")
    _sample_at(sd, "candidate_000300_02_20260508122616_44_000.png")
    # End-of-epoch samples use an `e`-prefixed step token.
    _sample_at(sd, "candidate_e000001_01_20260508122941_43_000.png")
    pairing = _pair_samples_with_prompts(sd, ["A", "B", "C"])
    assert len(pairing) == 4
    values = list(pairing.values())
    assert values.count("A") == 1
    assert values.count("B") == 2  # idx=1 appears in both step-300 and end-of-epoch
    assert values.count("C") == 1


def test_pair_samples_uses_sidecar_txt_when_present(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    img = _sample_at(sd, "candidate_000200_0_42.png")
    img.with_suffix(".txt").write_text("override prompt", encoding="utf-8")
    pairing = _pair_samples_with_prompts(sd, ["fallback"])
    assert pairing[img] == "override prompt"


def test_scorer_emits_in_dist_and_ood_components(tmp_path: Path):
    """When the caller flags an in-dist / OOD split, the scorer must
    emit `in_dist_score`, `ood_score`, and `generalization_gap` so the
    Pareto front (Agent #2's work) and the report can read them. Aggregate
    score is unchanged — still the mean across all judgements."""
    tfe = _write_tfevents(tmp_path / "ev", [0.4 - 0.001 * i for i in range(50)])
    sample_dir = tmp_path / "samples"; sample_dir.mkdir()
    # Two in-dist prompts, two OOD prompts. The trainer wrote one sample
    # per prompt index in the merged prompts file.
    for idx in range(4):
        _sample_at(sample_dir, f"candidate_000040_{idx}_42.png")

    # Judge: easy prompts get 8/10, hard (OOD) prompts get 4/10.
    class _SplitJudge(SampleJudge):
        def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
            score = 8.0 if prompt.startswith("easy") else 4.0
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=score, visual_quality=score,
                artifact_free=score, overall=score,
                raw_response="",
            )

    prompts = ["easy 0", "easy 1", "hard 0", "hard 1"]
    scorer = Scorer(sample_judge=_SplitJudge(), loss_weight=0.3, sample_weight=0.7)
    rep = scorer.score_run(
        tfevents_path=tfe, sample_dir=sample_dir,
        prompts=prompts, n_in_dist_prompts=2,
    )

    assert rep.disqualified is None
    assert rep.judge_report is not None
    # Lower-is-better mapping: 1 - mean/10
    assert rep.components["in_dist_score"] == pytest.approx(1.0 - 8.0 / 10.0)
    assert rep.components["ood_score"] == pytest.approx(1.0 - 4.0 / 10.0)
    assert rep.components["generalization_gap"] == pytest.approx(0.4)
    assert rep.components["in_dist_samples_judged"] == pytest.approx(2.0)
    assert rep.components["ood_samples_judged"] == pytest.approx(2.0)


def test_scorer_skips_ood_split_when_not_requested(tmp_path: Path):
    """When n_in_dist_prompts is None, behave as before — no split keys."""
    tfe = _write_tfevents(tmp_path / "ev", [0.4 - 0.001 * i for i in range(50)])
    sample_dir = tmp_path / "samples"; sample_dir.mkdir()
    _sample_at(sample_dir, "candidate_000040_0_42.png")
    scorer = Scorer(sample_judge=_FixedJudge(score=8.0), loss_weight=0.3, sample_weight=0.7)
    rep = scorer.score_run(
        tfevents_path=tfe, sample_dir=sample_dir, prompts=["a beach"],
    )
    assert "in_dist_score" not in rep.components
    assert "ood_score" not in rep.components
    assert "generalization_gap" not in rep.components


def test_disqualified_run_skips_judge(tmp_path: Path):
    """If loss says diverging, we shouldn't waste VLM budget on a doomed run."""
    tfe = _write_tfevents(tmp_path / "ev", [0.3 + 0.05 * i for i in range(40)])
    sample_dir = tmp_path / "samples"; sample_dir.mkdir()
    _sample_at(sample_dir, "candidate_000040_0_42.png")
    judge_calls: list = []

    class _CountingJudge(SampleJudge):
        def judge_image(self, image_path, prompt):
            judge_calls.append((image_path, prompt))
            return SampleJudgement(
                image_path=image_path, prompt=prompt,
                prompt_adherence=10, visual_quality=10, artifact_free=10,
                overall=10, raw_response="",
            )

    scorer = Scorer(sample_judge=_CountingJudge(), positive_slope_kill_threshold=0.01)
    rep = scorer.score_run(tfevents_path=tfe, sample_dir=sample_dir, prompts=["x"])
    assert rep.disqualified is not None
    assert judge_calls == []
