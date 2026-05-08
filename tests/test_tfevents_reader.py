"""Tests run against a tiny on-disk tfevents file we synthesize via TF summary
writer — that's the same surface accelerate's TensorBoardTracker writes through,
so behaviour matches a real musubi_tuner run."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

# Use the bundled tensorboard summary writer (no torch/accelerate dep)
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from bracket.frame import LossFrame
from bracket.tfevents_reader import TFEventsTail, scalar_tags_in


def _write_scalar(writer: EventFileWriter, tag: str, value: float, step: int, wall: float) -> None:
    summ = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
    writer.add_event(Event(wall_time=wall, step=step, summary=summ))


def _make_event_file(tmp: Path, n_steps: int = 50) -> str:
    writer = EventFileWriter(str(tmp), max_queue_size=1024, flush_secs=0)
    base = time.time()
    for step in range(1, n_steps + 1):
        # alternate high/low losses, same shape as the real chart
        loss = 0.305 if step % 2 == 0 else 0.365
        _write_scalar(writer, "loss/current", loss, step, base + step * 0.1)
        _write_scalar(writer, "lr/unet", 1e-6, step, base + step * 0.1)
    writer.close()
    # Find the event file
    files = sorted(p for p in tmp.glob("events.out.tfevents.*"))
    assert files, "no tfevents file written"
    return str(files[-1])


def test_scalar_tags_in_finds_loss_and_lr():
    with tempfile.TemporaryDirectory() as td:
        path = _make_event_file(Path(td), n_steps=10)
        tags = scalar_tags_in(path)
        assert "loss/current" in tags
        assert "lr/unet" in tags


def test_drain_once_emits_all_steps_in_order():
    with tempfile.TemporaryDirectory() as td:
        path = _make_event_file(Path(td), n_steps=50)
        seen: list[LossFrame] = []
        tail = TFEventsTail(event_path=path, on_frame=seen.append, ema_alpha=0.1)
        n = tail.drain_once()
        assert n == 50
        assert len(seen) == 50
        assert [f.step for f in seen] == list(range(1, 51))
        assert all(f.lr == pytest.approx(1e-6) for f in seen)
        assert all(f.timestep is None for f in seen)  # v0.1: no timestep info
        assert all(f.grad_norm is None for f in seen)


def test_drain_once_idempotent_on_no_new_data():
    with tempfile.TemporaryDirectory() as td:
        path = _make_event_file(Path(td), n_steps=10)
        seen: list[LossFrame] = []
        tail = TFEventsTail(event_path=path, on_frame=seen.append, ema_alpha=0.1)
        assert tail.drain_once() == 10
        # Second drain — no new events written, should emit zero new frames
        assert tail.drain_once() == 0
        assert len(seen) == 10


def test_smoothed_loss_dampens_oscillation():
    """End-to-end: alternating 0.305/0.365 input → smoothed should converge near
    the mean (0.335). This is the wedge value: a chart that's actually readable."""
    with tempfile.TemporaryDirectory() as td:
        path = _make_event_file(Path(td), n_steps=500)
        seen: list[LossFrame] = []
        tail = TFEventsTail(event_path=path, on_frame=seen.append, ema_alpha=0.05)
        tail.drain_once()
        last_smoothed = seen[-1].smoothed_loss
        assert 0.32 < last_smoothed < 0.35  # converged near mean


def test_missing_event_file_raises():
    with pytest.raises(FileNotFoundError):
        TFEventsTail(event_path="/nonexistent/path", on_frame=lambda f: None)


def test_callback_exception_does_not_break_iteration():
    """A misbehaving subscriber must not stop the tail or skip frames."""
    with tempfile.TemporaryDirectory() as td:
        path = _make_event_file(Path(td), n_steps=10)
        seen: list[LossFrame] = []

        def flaky(frame: LossFrame) -> None:
            if frame.step == 3:
                raise RuntimeError("subscriber blew up")
            seen.append(frame)

        tail = TFEventsTail(event_path=path, on_frame=flaky, ema_alpha=0.1)
        tail.drain_once()
        # Step 3 was dropped (callback raised), but all others made it
        assert [f.step for f in seen] == [1, 2, 4, 5, 6, 7, 8, 9, 10]
