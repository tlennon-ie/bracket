"""Scorer tests for the LTX-2 (ltx-trainer) stdout-log path.

LTX-2 produces no tfevents — the scorer must instead read the loss curve from
the run's stdout log (``loss_log_path``) and run the SAME divergence / slope /
NaN disqualification logic. It also writes validation samples as
``step_{step:06d}_{idx+1}.{mp4|png|wav}`` with a 1-based prompt index and NO
sidecar ``.txt`` — pairing must map those to the right prompt without
regressing the existing sd-scripts / musubi conventions.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from bracket.orchestrator.scorer import (
    Scorer,
    _extract_prompt_idx,
    _pair_samples_with_prompts,
)


def _write_ltx_log(tmp: Path, losses: list[float], *, step_every: int = 20) -> Path:
    """Write an LTX-2-style stdout log: one cadence line per logged step."""
    p = tmp / "stdout.log"
    lines = [
        "# bracket run ltx-test",
        "# cmd: ['python', 'train.py']",
        "#",
        "Loading LTX-2 checkpoint...",
    ]
    for i, loss in enumerate(losses, start=1):
        step = i * step_every
        lines.append(
            f"Step {step}/2000 - Loss: {loss:.4f}, LR: 1.00e-04, "
            f"Time/Step: 2.31s, Total Time: 0:0{i}:00"
        )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _sample_at(dir_: Path, name: str) -> Path:
    p = dir_ / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 16)
    return p


# --- (i) loss scoring from a stdout log ------------------------------------


def test_score_run_from_log_converging_is_not_disqualified(tmp_path: Path):
    losses = [0.5 - 0.003 * i for i in range(100)]
    log = _write_ltx_log(tmp_path, losses)
    rep = Scorer(positive_slope_kill_threshold=0.1).score_run(
        tfevents_path=None, loss_log_path=log, loss_source="stdout_log",
    )
    assert rep.disqualified is None
    assert math.isfinite(rep.score)
    assert rep.n_steps == 100
    assert rep.components["slope"] < 0  # going down
    assert "loss_only" in rep.components


def test_score_run_from_log_diverging_is_disqualified(tmp_path: Path):
    # Loss climbs steeply. Steps are spaced 20 apart (LTX-2 cadence), so the
    # per-step slope is climb_per_point / 20 — pick a climb that still trips
    # the kill threshold against the real (step-valued) x-axis.
    losses = [0.3 + 1.0 * i for i in range(40)]
    log = _write_ltx_log(tmp_path, losses)
    rep = Scorer(positive_slope_kill_threshold=0.01).score_run(
        tfevents_path=None, loss_log_path=log, loss_source="stdout_log",
    )
    assert rep.score == math.inf
    assert rep.disqualified is not None and rep.disqualified.startswith("diverging")


def test_score_run_from_log_nan_loss_disqualifies(tmp_path: Path):
    """A run whose loss goes NaN (gradient explosion) must be caught as
    ``nan_loss`` — NOT silently dropped (which would score it off the last
    finite frame) and NOT mislabelled ``empty_loss_log``. Guards the regex
    that captures ``nan``/``inf`` tokens from the stdout log."""
    losses = [0.5, 0.4, 0.3, float("nan")]
    log = _write_ltx_log(tmp_path, losses)
    rep = Scorer().score_run(
        tfevents_path=None, loss_log_path=log, loss_source="stdout_log",
    )
    assert rep.score == math.inf
    assert rep.disqualified == "nan_loss"


def test_score_run_no_loss_log_disqualifies(tmp_path: Path):
    rep = Scorer().score_run(
        tfevents_path=None, loss_log_path=None, loss_source="stdout_log",
    )
    assert rep.score == math.inf
    assert rep.disqualified == "no_loss_log"


def test_score_run_empty_loss_log_disqualifies(tmp_path: Path):
    p = tmp_path / "stdout.log"
    p.write_text("# header only\nno loss lines here\n", encoding="utf-8")
    rep = Scorer().score_run(
        tfevents_path=None, loss_log_path=p, loss_source="stdout_log",
    )
    assert rep.score == math.inf
    assert rep.disqualified == "empty_loss_log"


def test_score_run_tfevents_trainer_missing_events_keeps_no_tfevents(tmp_path: Path):
    """An existing tfevents trainer that crashed before writing any events
    (loss_source defaults to 'tfevents') must keep the historical
    ``no_tfevents`` DQ reason — NOT be relabelled ``empty_loss_log`` off the
    always-present run log. Regression guard for ledger/report consistency."""
    # A perfectly valid-looking LTX-style log exists on disk, but this is a
    # tfevents trainer, so the log must be ignored entirely.
    log = _write_ltx_log(tmp_path, [0.5 - 0.003 * i for i in range(50)])
    rep = Scorer().score_run(
        tfevents_path=None, loss_log_path=log,  # loss_source defaults to "tfevents"
    )
    assert rep.score == math.inf
    assert rep.disqualified == "no_tfevents"


def test_score_run_prefers_tfevents_when_both_present(tmp_path: Path):
    """When tfevents is present it wins; the log path is ignored."""
    import time as _time

    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    ev_dir = tmp_path / "ev"
    writer = EventFileWriter(str(ev_dir), max_queue_size=512, flush_secs=0)
    base = _time.time()
    for step, loss in enumerate([0.4 - 0.001 * i for i in range(50)], start=1):
        writer.add_event(Event(
            wall_time=base + step * 0.01, step=step,
            summary=Summary(value=[Summary.Value(tag="loss/current", simple_value=loss)]),
        ))
    writer.close()
    tfe = next(p for p in ev_dir.glob("events.out.tfevents.*"))
    # A diverging log that WOULD be disqualified if it were read.
    bad_log = _write_ltx_log(tmp_path, [0.3 + 0.05 * i for i in range(40)])
    rep = Scorer(positive_slope_kill_threshold=0.1).score_run(
        tfevents_path=tfe, loss_log_path=bad_log,
    )
    assert rep.disqualified is None
    assert math.isfinite(rep.score)


# --- (ii) LTX-2 1-based filename pairing -----------------------------------


def test_extract_prompt_idx_ltx2_step_format():
    # step_000100_3 → 1-based idx 3 → prompt index 2
    assert _extract_prompt_idx("step_000100_3") == 2
    assert _extract_prompt_idx("step_000020_1") == 0
    # Extracted video frame stem: step_000100_3_frame00 → still prompt index 2
    assert _extract_prompt_idx("step_000100_3_frame00") == 2


def test_pair_samples_ltx2_step_format(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    _sample_at(sd, "step_000100_1.png")  # 1-based idx 1 → prompt A
    _sample_at(sd, "step_000100_2.png")  # 1-based idx 2 → prompt B
    _sample_at(sd, "step_000100_3.png")  # 1-based idx 3 → prompt C
    pairing = _pair_samples_with_prompts(sd, ["A", "B", "C"])
    assert pairing[sd / "step_000100_1.png"] == "A"
    assert pairing[sd / "step_000100_2.png"] == "B"
    assert pairing[sd / "step_000100_3.png"] == "C"


def test_pair_samples_ltx2_extracted_frames(tmp_path: Path):
    """Frames extracted from LTX-2 videos carry the index in the parent
    video stem (``step_000100_3_frame00.png``) and must still pair."""
    sd = tmp_path / "samples"; sd.mkdir()
    frames_dir = sd / "_frames"; frames_dir.mkdir()
    _sample_at(frames_dir, "step_000100_2_frame00.png")
    _sample_at(frames_dir, "step_000100_2_frame01.png")
    _sample_at(frames_dir, "step_000100_3_frame00.png")
    pairing = _pair_samples_with_prompts(sd, ["A", "B", "C"])
    assert pairing[frames_dir / "step_000100_2_frame00.png"] == "B"
    assert pairing[frames_dir / "step_000100_2_frame01.png"] == "B"
    assert pairing[frames_dir / "step_000100_3_frame00.png"] == "C"


# --- (iii) no regression on existing conventions ---------------------------


def test_pair_samples_sdscripts_still_pairs(tmp_path: Path):
    sd = tmp_path / "samples"; sd.mkdir()
    _sample_at(sd, "candidate_000200_0_42.png")
    _sample_at(sd, "candidate_000200_1_42.png")
    _sample_at(sd, "candidate_000200_2_42.png")
    pairing = _pair_samples_with_prompts(sd, ["A", "B", "C"])
    assert len(pairing) == 3
    assert pairing[sd / "candidate_000200_0_42.png"] == "A"
    assert pairing[sd / "candidate_000200_2_42.png"] == "C"


def test_extract_prompt_idx_sdscripts_unchanged():
    # sd-scripts: <name>_<step>_<idx>_<seed> → idx is parts[-2], 0-based.
    assert _extract_prompt_idx("candidate_000200_0_42") == 0
    assert _extract_prompt_idx("candidate_000200_2_42") == 2
