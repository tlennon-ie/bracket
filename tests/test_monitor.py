"""Tests for the Monitor tab data extraction."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from bracket.ui.monitor import (
    build_snapshot,
    find_tfevents_in,
    latest_running_run,
    load_loss_series,
    parse_max_steps_from_log,
    score_history_rows,
    setup_status,
)


def _make_tfevents(d: Path, losses: list[float]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    w = EventFileWriter(str(d), max_queue_size=512, flush_secs=0)
    base = time.time()
    for step, loss in enumerate(losses, start=1):
        w.add_event(Event(
            wall_time=base + step * 0.01, step=step,
            summary=Summary(value=[Summary.Value(tag="loss/current", simple_value=loss)]),
        ))
    w.close()
    return next(p for p in d.glob("events.out.tfevents.*"))


def _write_ledger(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_setup_status_transitions(tmp_path: Path):
    assert setup_status([]) == "not started"
    assert setup_status([{"role": "setup", "exit_code": 0}]) == "done"
    assert setup_status([{"role": "setup", "exit_code": None}]) == "running"
    assert setup_status([{"role": "setup", "exit_code": 1}]) == "errored"
    # Mixed: one done + one running = running
    assert setup_status([
        {"role": "setup", "exit_code": 0},
        {"role": "setup", "exit_code": None},
    ]) == "running"


def test_latest_running_run_finds_ungelected_dir(tmp_path: Path):
    runs = tmp_path / "runs"
    a = runs / "cand-000-s0"; a.mkdir(parents=True)
    b = runs / "cand-001-s0"; b.mkdir(parents=True)
    # bump b's mtime so it's the newer one
    time.sleep(0.01)
    (b / "marker").write_text("x", encoding="utf-8")
    rows = [{"run_id": "cand-000-s0"}]   # 'a' already in ledger
    chosen = latest_running_run(tmp_path, rows)
    assert chosen is not None and chosen.name == "cand-001-s0"


def test_latest_running_run_falls_back_to_completed_run_when_no_inflight(tmp_path: Path):
    """When every run is in the ledger (no in-flight), we still want to
    plot loss for the most recent one — not return None and blank the chart."""
    runs = tmp_path / "runs"
    a = runs / "cand-000-s0"; a.mkdir(parents=True)
    rows = [{"run_id": "cand-000-s0"}]
    chosen = latest_running_run(tmp_path, rows)
    assert chosen is not None
    assert chosen.name == "cand-000-s0"


def test_latest_running_run_prefers_inflight_over_completed(tmp_path: Path):
    runs = tmp_path / "runs"
    done = runs / "cand-000-s0"; done.mkdir(parents=True)
    time.sleep(0.01)
    inflight = runs / "cand-001-s0"; inflight.mkdir(parents=True)
    rows = [{"run_id": "cand-000-s0"}]   # only 'done' in ledger
    chosen = latest_running_run(tmp_path, rows)
    assert chosen is not None and chosen.name == "cand-001-s0"


def test_gallery_finds_samples_in_singular_subdir(tmp_path: Path):
    """sd-scripts and musubi both save samples to <output_dir>/sample/ (singular).
    The gallery must look there, not at <run_dir>/samples/."""
    from bracket.ui.monitor import gallery_items
    runs = tmp_path / "runs" / "cand-000"
    sample_dir = runs / "output" / "sample"
    sample_dir.mkdir(parents=True)
    img = sample_dir / "candidate_000020_0_42.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    items = gallery_items(tmp_path)
    assert len(items) == 1
    assert items[0][0].endswith("candidate_000020_0_42.png")


def test_gallery_groups_returns_one_group_per_run_newest_first(tmp_path: Path):
    """Monitor tab uses gallery_groups() to render an accordion per run.

    Groups must be ordered newest-first by their freshest image's mtime,
    setup directories excluded, and runs with no samples skipped.
    """
    from bracket.ui.monitor import gallery_groups
    runs = tmp_path / "runs"
    # older run
    a = runs / "cand-000-s0" / "output" / "sample"; a.mkdir(parents=True)
    (a / "candidate_000300_00_20260508120000_42_000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # newer run with two images
    time.sleep(0.02)
    b = runs / "cand-001-s0" / "output" / "sample"; b.mkdir(parents=True)
    (b / "candidate_000300_00_20260508121000_43_000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (b / "candidate_000300_01_20260508121017_43_000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # setup dir is filtered out, even if it had images
    s = runs / "setup" / "output" / "sample"; s.mkdir(parents=True)
    (s / "candidate_000001_00_20260508110000_99_000.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    # run dir without any samples is skipped
    (runs / "cand-002-s0").mkdir(parents=True)

    groups = gallery_groups(tmp_path)
    assert [g.run_id for g in groups] == ["cand-001-s0", "cand-000-s0"]
    assert len(groups[0].items) == 2
    assert len(groups[1].items) == 1


def test_load_loss_series_returns_smoothed_and_raw(tmp_path: Path):
    log_dir = tmp_path / "logs" / "run_x"
    tfe = _make_tfevents(log_dir, [0.5 - 0.001 * i for i in range(50)])
    series = load_loss_series(tfe)
    assert series is not None
    assert len(series.steps) == 50
    assert len(series.raw) == 50
    assert len(series.smoothed) == 50
    assert len(series.wall_times) == 50
    # wall_times must be monotonic — used to compute it/s
    for a, b in zip(series.wall_times, series.wall_times[1:]):
        assert b >= a


def test_compute_steps_per_sec_returns_average_rate():
    from bracket.ui.monitor import LossSeries, compute_steps_per_sec

    # 10 steps over 5 seconds = 2 it/s
    s = LossSeries(
        steps=list(range(10)),
        raw=[0.5] * 10,
        smoothed=[0.5] * 10,
        wall_times=[i * 0.5 for i in range(10)],
    )
    rate = compute_steps_per_sec(s)
    assert rate is not None
    assert abs(rate - 2.0) < 0.01


def test_compute_steps_per_sec_returns_none_without_wall_times():
    from bracket.ui.monitor import LossSeries, compute_steps_per_sec

    s = LossSeries(steps=[0, 1, 2], raw=[0.5] * 3, smoothed=[0.5] * 3)
    assert compute_steps_per_sec(s) is None


def test_compute_steps_per_sec_returns_none_for_single_sample():
    from bracket.ui.monitor import LossSeries, compute_steps_per_sec

    s = LossSeries(
        steps=[0], raw=[0.5], smoothed=[0.5], wall_times=[100.0],
    )
    assert compute_steps_per_sec(s) is None


def test_compute_steps_per_sec_handles_zero_dt():
    """If somehow all sampled wall_times are identical (e.g. tfevents bug),
    we must return None rather than divide by zero."""
    from bracket.ui.monitor import LossSeries, compute_steps_per_sec

    s = LossSeries(
        steps=[0, 1, 2],
        raw=[0.5, 0.5, 0.5],
        smoothed=[0.5, 0.5, 0.5],
        wall_times=[100.0, 100.0, 100.0],
    )
    assert compute_steps_per_sec(s) is None


def test_parse_max_steps_from_log(tmp_path: Path):
    run = tmp_path / "run"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "stdout.log").write_text(
        "# bracket run cand\n"
        "# cwd: I:/AI/musubi-tuner/sd-scripts\n"
        "# cmd: ['python.exe', 'sdxl_train_network.py', '--max_train_steps', '300']\n",
        encoding="utf-8",
    )
    assert parse_max_steps_from_log(run) == 300


def test_score_history_includes_curated_role(tmp_path: Path):
    """Curated warm-start runs must appear in the score-history table and
    the completed-runs counter — they're real training trials. Earlier
    versions silently dropped them, leaving the dashboard stuck at e.g.
    "5/9" while showing 4 ghost training runs."""
    rows = [
        {"role": "setup", "run_id": "setup-000", "exit_code": 0},
        {"role": "baseline", "run_id": "baseline-000", "score": 0.5,
         "score_components": {"final_smoothed": 0.49, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
        {"role": "curated", "run_id": "cur-000-s0", "score": 0.45,
         "score_components": {"final_smoothed": 0.44, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
        {"role": "curated", "run_id": "cur-001-s0", "score": 0.42,
         "score_components": {"final_smoothed": 0.41, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
        {"role": "candidate", "run_id": "cand-002-s0", "score": 0.40,
         "score_components": {"final_smoothed": 0.39, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
    ]
    history = score_history_rows(rows)
    assert len(history) == 4
    roles = sorted(h.role for h in history)
    assert roles == ["baseline", "candidate", "curated", "curated"]


def test_build_snapshot_counts_curated_in_completed(tmp_path: Path):
    """build_snapshot.completed_runs must include curated runs so progress
    reaches 100% when all budget slots (curated + candidate) are filled."""
    out = tmp_path / "session"
    runs = out / "runs"
    runs.mkdir(parents=True)
    _write_ledger(out / "ledger.jsonl", [
        {"role": "setup", "run_id": "setup-000", "exit_code": 0},
        {"role": "baseline", "run_id": "baseline-000-s0", "score": 0.5,
         "score_components": {"final_smoothed": 0.49, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
        {"role": "curated", "run_id": "cur-000-s0", "score": 0.45,
         "score_components": {"final_smoothed": 0.44, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
        {"role": "curated", "run_id": "cur-001-s0", "score": 0.43,
         "score_components": {"final_smoothed": 0.42, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
        {"role": "candidate", "run_id": "cand-002-s0", "score": 0.40,
         "score_components": {"final_smoothed": 0.39, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0},
    ])
    snap = build_snapshot(out, total_runs_target=4)
    assert snap.completed_runs == 4
    assert snap.progress_pct == 100.0
    assert snap.session_done is True


def test_build_snapshot_judge_summary_when_not_configured(tmp_path: Path):
    """If no run has a populated judge_report, the snapshot's judge_summary
    must explicitly say so — silence used to make users think LMStudio was
    being called when it wasn't."""
    out = tmp_path / "session"
    (out / "runs").mkdir(parents=True)
    _write_ledger(out / "ledger.jsonl", [
        {"role": "baseline", "run_id": "baseline-000-s0", "score": 0.5,
         "score_components": {"final_smoothed": 0.49, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0, "judge_report": None},
    ])
    snap = build_snapshot(out, total_runs_target=1)
    assert "not configured" in snap.judge_summary
    assert "loss only" in snap.judge_summary


def test_build_snapshot_judge_summary_when_active(tmp_path: Path):
    out = tmp_path / "session"
    (out / "runs").mkdir(parents=True)
    _write_ledger(out / "ledger.jsonl", [
        {"role": "baseline", "run_id": "baseline-000-s0", "score": 0.5,
         "score_components": {"final_smoothed": 0.49, "slope": -0.01},
         "n_steps": 100, "duration_s": 60.0,
         "judge_report": {"n_images": 10, "n_failed": 1, "mean_overall": 8.5,
                          "mean_prompt_adherence": 8.0, "mean_visual_quality": 8.7,
                          "mean_artifact_free": 9.0}},
    ])
    snap = build_snapshot(out, total_runs_target=1)
    assert "LMStudio" in snap.judge_summary
    assert "10 images" in snap.judge_summary
    assert "8.50" in snap.judge_summary


def test_build_snapshot_judge_summary_when_configured_but_no_scored_row(tmp_path: Path):
    """User configured the judge but stopped before any run finished.
    Must NOT fall back to the misleading "not configured" message."""
    out = tmp_path / "session"
    (out / "runs").mkdir(parents=True)
    _write_ledger(out / "ledger.jsonl", [
        {"role": "setup", "run_id": "setup-000", "exit_code": 0},
    ])
    snap = build_snapshot(out, total_runs_target=1, judge_configured=True)
    assert "configured" in snap.judge_summary
    assert "not configured" not in snap.judge_summary
    assert "waiting for the first run" in snap.judge_summary


def test_build_snapshot_judge_summary_killed_before_first_score(tmp_path: Path):
    """Same case but with a row that timed out before scoring (judge_report=None)."""
    out = tmp_path / "session"
    (out / "runs").mkdir(parents=True)
    _write_ledger(out / "ledger.jsonl", [
        {"role": "baseline", "run_id": "baseline-000-s0", "score": None,
         "score_components": {}, "n_steps": 3, "duration_s": 1800.0,
         "judge_report": None, "killed_by_timeout": True},
    ])
    snap = build_snapshot(out, total_runs_target=1, judge_configured=True)
    assert "not configured" not in snap.judge_summary
    assert "configured" in snap.judge_summary


def test_score_history_filters_setup_rows(tmp_path: Path):
    rows = [
        {"role": "setup", "run_id": "setup-000", "exit_code": 0},
        {"role": "baseline", "run_id": "baseline-000", "score": 0.5,
         "score_components": {"final_smoothed": 0.48, "slope": -0.02},
         "n_steps": 100, "duration_s": 60.0},
        {"role": "candidate", "run_id": "cand-000", "score": 0.4,
         "score_components": {"final_smoothed": 0.39, "slope": -0.01},
         "n_steps": 100, "duration_s": 65.0},
    ]
    history = score_history_rows(rows)
    assert len(history) == 2
    assert history[0].run_id == "baseline-000"
    assert history[1].score == 0.4


def test_build_snapshot_end_to_end(tmp_path: Path):
    out = tmp_path / "session"
    runs = out / "runs"
    runs.mkdir(parents=True)
    # Setup row + completed baseline + in-progress candidate
    _write_ledger(out / "ledger.jsonl", [
        {"role": "setup", "run_id": "setup-000", "exit_code": 0},
        {"role": "baseline", "run_id": "baseline-000-s0", "score": 0.42, "config_id": "abc",
         "config": {"learning_rate": 1e-4}, "n_steps": 100, "duration_s": 50.0,
         "score_components": {"final_smoothed": 0.41, "slope": -0.01}},
    ])
    # In-progress candidate dir with tfevents
    cand = runs / "cand-000-s0"
    _make_tfevents(cand / "logs" / "run_x", [0.5 - 0.001 * i for i in range(30)])
    (cand / "logs" / "stdout.log").parent.mkdir(parents=True, exist_ok=True)
    (cand / "logs" / "stdout.log").write_text(
        "# cmd: ['python', 'sdxl_train_network.py', '--max_train_steps', '300']\n",
        encoding="utf-8",
    )
    snap = build_snapshot(out, total_runs_target=10)
    assert snap.setup_status == "done"
    assert snap.completed_runs == 1
    assert snap.current_run_id == "cand-000-s0"
    assert snap.current_run_max_steps == 300
    assert snap.current_run_steps_done == 30
    assert snap.current_loss is not None
    assert len(snap.current_loss.steps) == 30
    assert len(snap.score_history) == 1


def test_build_snapshot_recomputes_smoothed_when_alpha_changes(tmp_path: Path):
    """Smoothing slider in the Monitor tab passes through to build_snapshot.

    With alpha=1.0 the smoothed series should equal raw; with alpha=0.05
    the series should be visibly damped. Different alphas must produce
    different curves so the slider actually does something live.
    """
    out = tmp_path / "session"
    runs = out / "runs"
    runs.mkdir(parents=True)
    cand = runs / "cand-000-s0"
    _make_tfevents(cand / "logs" / "run_x", [0.5 + 0.3 * (i % 2) for i in range(40)])

    snap_raw = build_snapshot(out, total_runs_target=1, ema_alpha=1.0)
    snap_smooth = build_snapshot(out, total_runs_target=1, ema_alpha=0.05)
    assert snap_raw.current_loss is not None
    assert snap_smooth.current_loss is not None
    # alpha=1.0 → no damping → smoothed exactly tracks raw
    assert snap_raw.current_loss.smoothed == snap_raw.current_loss.raw
    # alpha=0.05 → heavy damping → smoothed should differ on at least one step
    assert snap_smooth.current_loss.smoothed != snap_smooth.current_loss.raw


def test_build_snapshot_when_idle(tmp_path: Path):
    snap = build_snapshot(tmp_path / "no_session", total_runs_target=0)
    assert snap.completed_runs == 0
    assert snap.current_loss is None
    assert snap.score_history == []
