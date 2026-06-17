"""Tests for the stdout-log loss reader used by the LTX-2 (ltx-trainer) backend.

LTX-2 does NOT write tfevents — it logs one line every 20 steps to stdout in
the form::

    Step 20/2000 - Loss: 0.1234, LR: 1.00e-04, Time/Step: 2.31s, Total Time: 0:01:23

Bracket's runner redirects that stdout to a per-run log file (with leading
``#`` header lines). The reader must pick out only the matching lines and
ignore everything else.
"""
from __future__ import annotations

from pathlib import Path

from bracket.log_loss_reader import LogLossReader, parse_log_loss


def _write_log(tmp: Path, lines: list[str]) -> Path:
    p = tmp / "stdout.log"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_reader_parses_step_loss_lr_lines(tmp_path: Path):
    log = _write_log(
        tmp_path,
        [
            "# bracket run abc",
            "# cwd: /trainers/ltx",
            "# cmd: ['python', 'train.py']",
            "#",
            "Loading model weights...",
            "Step 20/2000 - Loss: 0.5000, LR: 1.00e-04, Time/Step: 2.31s, Total Time: 0:01:23",
            "some arbitrary trainer chatter that must be ignored",
            "Step 40/2000 - Loss: 0.4000, LR: 9.50e-05, Time/Step: 2.30s, Total Time: 0:02:46",
            "Step 60/2000 - Loss: 0.3000, LR: 9.00e-05, Time/Step: 2.29s, Total Time: 0:04:09",
            "Validation done.",
        ],
    )
    frames = LogLossReader(log).read_all()
    assert len(frames) == 3
    assert [f.step for f in frames] == [20, 40, 60]
    assert [round(f.raw_loss, 4) for f in frames] == [0.5, 0.4, 0.3]
    # LR parsed from the scientific-notation field.
    assert frames[0].lr == 1.0e-4
    assert frames[1].lr == 9.5e-5
    # EMA smoothed_loss is populated and seeded with the first raw value.
    assert frames[0].smoothed_loss == 0.5
    assert all(f.smoothed_loss is not None for f in frames)
    # tfevents-only fields stay None.
    assert all(f.timestep is None for f in frames)
    assert all(f.timestep_bucket is None for f in frames)
    assert all(f.grad_norm is None for f in frames)
    # wall_time is display-only but must be a float.
    assert all(isinstance(f.wall_time, float) for f in frames)


def test_reader_steps_are_monotonic_and_deduplicated(tmp_path: Path):
    log = _write_log(
        tmp_path,
        [
            "Step 40/2000 - Loss: 0.4000, LR: 1.00e-04, Time/Step: 2.0s, Total Time: 0:01:00",
            "Step 20/2000 - Loss: 0.5000, LR: 1.00e-04, Time/Step: 2.0s, Total Time: 0:00:40",
            "Step 40/2000 - Loss: 0.4200, LR: 1.00e-04, Time/Step: 2.0s, Total Time: 0:01:00",
            "Step 60/2000 - Loss: 0.3000, LR: 1.00e-04, Time/Step: 2.0s, Total Time: 0:01:20",
        ],
    )
    frames = parse_log_loss(log)
    steps = [f.step for f in frames]
    assert steps == [20, 40, 60]            # sorted ascending
    assert steps == sorted(set(steps))      # deduplicated


def test_reader_handles_missing_lr(tmp_path: Path):
    log = _write_log(
        tmp_path,
        [
            "Step 20/2000 - Loss: 0.5000, Time/Step: 2.0s",
            "Step 40/2000 - Loss: 0.4000, Time/Step: 2.0s",
        ],
    )
    frames = LogLossReader(log).read_all()
    assert len(frames) == 2
    assert all(f.lr is None for f in frames)


def test_reader_empty_or_junk_only_returns_no_frames(tmp_path: Path):
    log = _write_log(
        tmp_path,
        [
            "# header only",
            "nothing matches here",
            "Step but not a real line",
        ],
    )
    assert LogLossReader(log).read_all() == []


def test_reader_missing_file_returns_no_frames(tmp_path: Path):
    assert LogLossReader(tmp_path / "does_not_exist.log").read_all() == []


def test_reader_captures_nan_and_inf_loss(tmp_path: Path):
    """PyTorch trainers print ``Loss: nan`` / ``Loss: inf`` on gradient
    explosion (``f"{float('nan'):.4f}"`` -> ``"nan"``). The reader must
    capture them so the scorer's NaN/Inf disqualification fires, instead of
    dropping the line and scoring the run off its last finite frame."""
    import math

    log = _write_log(
        tmp_path,
        [
            "Step 20/2000 - Loss: 0.5000, LR: 1.00e-04, Time/Step: 2.0s",
            "Step 40/2000 - Loss: nan, LR: 1.00e-04, Time/Step: 2.0s",
            "Step 60/2000 - Loss: inf, LR: 1.00e-04, Time/Step: 2.0s",
            "Step 80/2000 - Loss: -inf, LR: 1.00e-04, Time/Step: 2.0s",
        ],
    )
    frames = LogLossReader(log).read_all()
    assert [f.step for f in frames] == [20, 40, 60, 80]
    assert frames[0].raw_loss == 0.5
    assert math.isnan(frames[1].raw_loss)
    assert math.isinf(frames[2].raw_loss) and frames[2].raw_loss > 0
    assert math.isinf(frames[3].raw_loss) and frames[3].raw_loss < 0
