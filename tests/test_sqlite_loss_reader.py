"""Tests for the SQLite loss reader used by the ai-toolkit (ostris) backend.

ai-toolkit writes NO tfevents. With ``logging.use_ui_logger: true`` its
``UILogger`` (toolkit/logging_aitk.py) writes a structured SQLite ``loss_log.db``
in WAL mode at ``<training_folder>/<name>/loss_log.db``. Schema::

    steps(step INTEGER PRIMARY KEY, wall_time REAL NOT NULL)
    metrics(step INTEGER, key TEXT, value_real REAL, value_text TEXT,
            PRIMARY KEY(step, key))
    metric_keys(key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER)

Loss is logged under ``key = 'loss'``; learning rate (if present) under
``key = 'lr'`` or ``'learning_rate'``. The DB is closed when the run finishes,
so the reader opens it read-only and post-hoc.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

from bracket.sqlite_loss_reader import SqliteLossReader, parse_sqlite_loss


def _build_db(
    path: Path,
    rows: Sequence[tuple[int, float, Optional[float]]],
    *,
    lr_key: str = "lr",
    extra_metric_keys: Sequence[str] = (),
) -> Path:
    """Create a synthetic loss_log.db matching the UILogger schema.

    ``rows`` is a list of ``(step, loss, lr_or_None)`` tuples. ``lr`` rows are
    only written when the lr value is not None. ``extra_metric_keys`` lets a
    test add unrelated metric rows (e.g. ``'lr_text'``) to prove the reader
    ignores everything but loss/lr.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE steps (step INTEGER PRIMARY KEY, wall_time REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE metrics ("
            "step INTEGER, key TEXT, value_real REAL, value_text TEXT, "
            "PRIMARY KEY(step, key))"
        )
        conn.execute(
            "CREATE TABLE metric_keys ("
            "key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER)"
        )
        for step, loss, lr in rows:
            conn.execute(
                "INSERT INTO steps (step, wall_time) VALUES (?, ?)",
                (step, 1000.0 + step),
            )
            conn.execute(
                "INSERT INTO metrics (step, key, value_real, value_text) "
                "VALUES (?, 'loss', ?, NULL)",
                (step, loss),
            )
            if lr is not None:
                conn.execute(
                    "INSERT INTO metrics (step, key, value_real, value_text) "
                    "VALUES (?, ?, ?, NULL)",
                    (step, lr_key, lr),
                )
            for extra in extra_metric_keys:
                conn.execute(
                    "INSERT INTO metrics (step, key, value_real, value_text) "
                    "VALUES (?, ?, ?, NULL)",
                    (step, extra, 0.0),
                )
        conn.commit()
    finally:
        conn.close()
    return path


def test_reader_parses_loss_and_lr_rows(tmp_path: Path):
    db = _build_db(
        tmp_path / "loss_log.db",
        [(20, 0.5, 1.0e-4), (40, 0.4, 9.5e-5), (60, 0.3, 9.0e-5)],
    )
    frames = SqliteLossReader(db).read_all()
    assert len(frames) == 3
    assert [f.step for f in frames] == [20, 40, 60]
    assert [round(f.raw_loss, 4) for f in frames] == [0.5, 0.4, 0.3]
    # lr mapped from the 'lr' metric key.
    assert frames[0].lr == 1.0e-4
    assert frames[1].lr == 9.5e-5
    # wall_time carried through from the steps table.
    assert frames[0].wall_time == 1020.0
    # EMA smoothed_loss is populated and seeded with the first raw value.
    assert frames[0].smoothed_loss == 0.5
    assert all(f.smoothed_loss is not None for f in frames)
    # tfevents-only fields stay None.
    assert all(f.timestep is None for f in frames)
    assert all(f.timestep_bucket is None for f in frames)
    assert all(f.grad_norm is None for f in frames)


def test_reader_returns_frames_sorted_ascending(tmp_path: Path):
    # Insert out of order; reader must return ascending by step.
    db = _build_db(
        tmp_path / "loss_log.db",
        [(60, 0.3, None), (20, 0.5, None), (40, 0.4, None)],
    )
    frames = parse_sqlite_loss(db)
    steps = [f.step for f in frames]
    assert steps == [20, 40, 60]
    assert steps == sorted(steps)


def test_reader_handles_learning_rate_key(tmp_path: Path):
    """lr may be logged under 'learning_rate' instead of 'lr'."""
    db = _build_db(
        tmp_path / "loss_log.db",
        [(20, 0.5, 1.0e-4), (40, 0.4, 9.0e-5)],
        lr_key="learning_rate",
    )
    frames = parse_sqlite_loss(db)
    assert frames[0].lr == 1.0e-4
    assert frames[1].lr == 9.0e-5


def test_reader_lr_absent_is_none(tmp_path: Path):
    db = _build_db(
        tmp_path / "loss_log.db",
        [(20, 0.5, None), (40, 0.4, None)],
    )
    frames = parse_sqlite_loss(db)
    assert len(frames) == 2
    assert all(f.lr is None for f in frames)


def test_reader_ignores_non_loss_metric_rows(tmp_path: Path):
    """Only 'loss' rows produce frames; unrelated metric keys are skipped."""
    db = _build_db(
        tmp_path / "loss_log.db",
        [(20, 0.5, 1.0e-4), (40, 0.4, 9.0e-5)],
        extra_metric_keys=("grad_norm", "epoch"),
    )
    frames = parse_sqlite_loss(db)
    assert [f.step for f in frames] == [20, 40]
    # The extra keys must not have leaked into raw_loss.
    assert [round(f.raw_loss, 4) for f in frames] == [0.5, 0.4]


def test_reader_missing_db_returns_no_frames(tmp_path: Path):
    assert SqliteLossReader(tmp_path / "does_not_exist.db").read_all() == []
    assert parse_sqlite_loss(tmp_path / "does_not_exist.db") == []


def test_reader_db_with_no_loss_rows_returns_empty(tmp_path: Path):
    # Steps + metric_keys exist but no 'loss' metric rows.
    db = _build_db(tmp_path / "loss_log.db", [])
    # Add a stray non-loss metric so the metrics table isn't empty either.
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("INSERT INTO steps (step, wall_time) VALUES (10, 1010.0)")
        conn.execute(
            "INSERT INTO metrics (step, key, value_real, value_text) "
            "VALUES (10, 'epoch', 1.0, NULL)"
        )
        conn.commit()
    finally:
        conn.close()
    assert parse_sqlite_loss(db) == []


def test_reader_unopenable_db_returns_no_frames(tmp_path: Path):
    # A non-SQLite file at the db path must not raise — return [].
    bogus = tmp_path / "loss_log.db"
    bogus.write_text("this is not a sqlite database", encoding="utf-8")
    assert parse_sqlite_loss(bogus) == []


def test_reader_propagates_nan_loss(tmp_path: Path):
    """A NaN loss row (gradient explosion) must propagate so the scorer's
    NaN disqualification fires, instead of being dropped."""
    db = _build_db(
        tmp_path / "loss_log.db",
        [(20, 0.5, 1.0e-4), (40, float("nan"), 1.0e-4)],
    )
    frames = parse_sqlite_loss(db)
    assert [f.step for f in frames] == [20, 40]
    assert frames[0].raw_loss == 0.5
    assert math.isnan(frames[1].raw_loss)
