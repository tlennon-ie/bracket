"""Scorer tests — synthesize tfevents files like the v0.1 reader tests do."""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from bracket.orchestrator.scorer import Scorer


def _write(writer: EventFileWriter, tag: str, value: float, step: int, wall: float) -> None:
    writer.add_event(Event(
        wall_time=wall, step=step,
        summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
    ))


def _write_tfevents(tmp: Path, losses: list[float]) -> Path:
    writer = EventFileWriter(str(tmp), max_queue_size=1024, flush_secs=0)
    base = time.time()
    for step, loss in enumerate(losses, start=1):
        _write(writer, "loss/current", loss, step, base + step * 0.01)
        _write(writer, "lr/unet", 1e-4, step, base + step * 0.01)
    writer.close()
    files = sorted(p for p in tmp.glob("events.out.tfevents.*"))
    assert files
    return files[-1]


def test_scorer_no_tfevents_disqualifies():
    s = Scorer().score_tfevents(None)
    assert s.score == math.inf
    assert s.disqualified == "no_tfevents"


def test_scorer_empty_tfevents_disqualifies(tmp_path: Path):
    # Create an empty events file with no scalar summaries
    writer = EventFileWriter(str(tmp_path), max_queue_size=1024, flush_secs=0)
    writer.close()
    files = sorted(p for p in tmp_path.glob("events.out.tfevents.*"))
    s = Scorer().score_tfevents(files[-1])
    assert s.score == math.inf
    assert s.disqualified == "empty_tfevents"


def test_scorer_converging_run_gets_finite_score(tmp_path: Path):
    # Loss decays from 0.5 to ~0.2 over 100 steps with noise
    import random as _rng
    rng = _rng.Random(0)
    losses = [0.5 - 0.003 * i + rng.uniform(-0.02, 0.02) for i in range(100)]
    path = _write_tfevents(tmp_path, losses)
    s = Scorer(positive_slope_kill_threshold=0.1).score_tfevents(path)
    assert s.disqualified is None
    assert math.isfinite(s.score)
    assert s.components["slope"] < 0  # going down


def test_scorer_diverging_run_disqualified(tmp_path: Path):
    # Loss climbs steadily — orchestrator should kill it
    losses = [0.3 + 0.05 * i for i in range(40)]
    path = _write_tfevents(tmp_path, losses)
    s = Scorer(positive_slope_kill_threshold=0.01).score_tfevents(path)
    assert s.score == math.inf
    assert s.disqualified is not None and s.disqualified.startswith("diverging")


def _write_tfevents_with_grad_norm(
    tmp: Path, losses: list[float], grad_norms: list[float],
) -> Path:
    """Synthesize an event file where each step has both a loss scalar
    AND a ``norm/avg_grad_norm`` scalar — emulates the sd-scripts LoRA
    flow with mean_grad_norm enabled."""
    assert len(losses) == len(grad_norms)
    writer = EventFileWriter(str(tmp), max_queue_size=1024, flush_secs=0)
    base = time.time()
    for step, (loss, gn) in enumerate(zip(losses, grad_norms), start=1):
        _write(writer, "loss/current", loss, step, base + step * 0.01)
        _write(writer, "lr/unet", 1e-4, step, base + step * 0.01)
        _write(writer, "norm/avg_grad_norm", gn, step, base + step * 0.01)
    writer.close()
    files = sorted(p for p in tmp.glob("events.out.tfevents.*"))
    assert files
    return files[-1]


def test_scorer_emits_grad_norm_components_when_logged(tmp_path: Path):
    losses = [0.4 - 0.001 * i for i in range(100)]
    grad_norms = [0.5 + 0.01 * (i % 5) for i in range(100)]
    path = _write_tfevents_with_grad_norm(tmp_path, losses, grad_norms)
    s = Scorer(positive_slope_kill_threshold=0.1).score_tfevents(path)
    assert s.disqualified is None
    assert "grad_norm_p99" in s.components
    assert "grad_norm_oscillation" in s.components
    # P99 of {0.5..0.54} is well below the 100.0 explosion cap.
    assert s.components["grad_norm_p99"] < 1.0
    # Oscillation is std/mean over the last 25%; small for bounded grad norms.
    assert s.components["grad_norm_oscillation"] >= 0


def test_scorer_disqualifies_on_gradient_explosion(tmp_path: Path):
    """A grad norm that blows past the hard cap means the rest of the
    loss curve is unreliable; the run must be tagged
    ``gradient_explosion`` even when the loss itself trends downward."""
    losses = [0.4 - 0.001 * i for i in range(60)]
    grad_norms = [0.5] * 58 + [500.0, 500.0]  # P99 > 100
    path = _write_tfevents_with_grad_norm(tmp_path, losses, grad_norms)
    s = Scorer(positive_slope_kill_threshold=0.1).score_tfevents(path)
    assert s.score == math.inf
    assert s.disqualified == "gradient_explosion"
    assert s.components["grad_norm_p99"] > 100.0


def test_scorer_disqualifies_on_nan_grad_norm(tmp_path: Path):
    """NaN/inf in the grad-norm stream means autograd produced
    non-finite gradients somewhere — the run is dead regardless of
    what the smoothed loss says."""
    losses = [0.4 - 0.001 * i for i in range(40)]
    grad_norms = [0.5] * 38 + [math.nan, math.inf]
    path = _write_tfevents_with_grad_norm(tmp_path, losses, grad_norms)
    s = Scorer(positive_slope_kill_threshold=0.1).score_tfevents(path)
    assert s.score == math.inf
    assert s.disqualified == "gradient_explosion"


def test_scorer_lower_final_loss_scores_better(tmp_path: Path):
    """Among two non-divergent runs, lower final smoothed loss → lower score."""
    a_path = _write_tfevents(
        tmp_path / "a", [0.5 - 0.001 * i for i in range(100)],
    )
    b_path = _write_tfevents(
        tmp_path / "b", [0.5 - 0.003 * i for i in range(100)],
    )
    sa = Scorer().score_tfevents(a_path)
    sb = Scorer().score_tfevents(b_path)
    assert sb.score < sa.score
