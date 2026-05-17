"""Tests for bracket.orchestrator.history — a SQLite-backed remembered-
baselines store. Each test uses a tmp_path-scoped DB so the user's real
~/.cache/bracket/history.db is never touched."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from bracket.orchestrator.history import (
    lookup_warmstart_configs,
    record_session_winner,
)


def test_lookup_returns_empty_when_db_missing(tmp_path: Path):
    db = tmp_path / "history.db"
    out = lookup_warmstart_configs("any", "any", db_path=db)
    assert out == []


def test_record_and_lookup_roundtrip(tmp_path: Path):
    db = tmp_path / "history.db"
    record_session_winner(
        trainer_name="sdxl", search_space_name="sdxl-lora",
        best_config={"lr": 1e-4, "batch": 2}, best_score=0.42,
        db_path=db,
    )
    out = lookup_warmstart_configs("sdxl", "sdxl-lora", db_path=db)
    assert out == [{"lr": 1e-4, "batch": 2}]


def test_lookup_returns_top_n_by_score(tmp_path: Path):
    db = tmp_path / "history.db"
    # Insert in shuffled score order; lookup must return them sorted.
    for cfg, score in [
        ({"lr": 1e-3}, 0.50),
        ({"lr": 2e-4}, 0.30),
        ({"lr": 5e-4}, 0.40),
        ({"lr": 1e-4}, 0.20),
        ({"lr": 3e-4}, 0.35),
    ]:
        record_session_winner(
            trainer_name="sdxl", search_space_name="sdxl-lora",
            best_config=cfg, best_score=score, db_path=db,
        )
    out = lookup_warmstart_configs("sdxl", "sdxl-lora", n=3, db_path=db)
    assert out == [{"lr": 1e-4}, {"lr": 2e-4}, {"lr": 3e-4}]


def test_lookup_respects_trainer_and_space_scoping(tmp_path: Path):
    """A history entry written for trainer A / space X must NOT show up
    when querying trainer B (different trainer entirely) — comparing
    scores across trainers is meaningless."""
    db = tmp_path / "history.db"
    record_session_winner(
        trainer_name="sdxl", search_space_name="sdxl-lora",
        best_config={"lr": 1e-4}, best_score=0.1, db_path=db,
    )
    record_session_winner(
        trainer_name="zimage_lora", search_space_name="zimage-lora",
        best_config={"lr": 5e-5}, best_score=0.2, db_path=db,
    )
    assert lookup_warmstart_configs("sdxl", "sdxl-lora", db_path=db) == [{"lr": 1e-4}]
    assert lookup_warmstart_configs("zimage_lora", "zimage-lora", db_path=db) == [
        {"lr": 5e-5},
    ]


def test_lookup_deduplicates_identical_configs(tmp_path: Path):
    """Two sessions that happened to land on the same winning config
    shouldn't fill the warm-start with duplicates — that wastes budget."""
    db = tmp_path / "history.db"
    for score in [0.25, 0.30, 0.22]:
        record_session_winner(
            trainer_name="sdxl", search_space_name="sdxl-lora",
            best_config={"lr": 1e-4, "batch": 2}, best_score=score, db_path=db,
        )
    record_session_winner(
        trainer_name="sdxl", search_space_name="sdxl-lora",
        best_config={"lr": 2e-4, "batch": 4}, best_score=0.40, db_path=db,
    )
    out = lookup_warmstart_configs("sdxl", "sdxl-lora", n=3, db_path=db)
    assert out == [{"lr": 1e-4, "batch": 2}, {"lr": 2e-4, "batch": 4}]


def test_lookup_swallows_corrupt_db_errors(tmp_path: Path):
    """A corrupt DB file must not crash the orchestrator — warm-start
    is a nice-to-have, not load bearing."""
    db = tmp_path / "history.db"
    db.write_bytes(b"this is not a sqlite file at all")
    out = lookup_warmstart_configs("sdxl", "sdxl-lora", db_path=db)
    assert out == []


def test_record_failure_does_not_raise(tmp_path: Path, monkeypatch):
    """A flaky filesystem at record time must be swallowed; the session
    that just completed shouldn't fail at the finish line."""
    db = tmp_path / "subdir" / "history.db"

    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("disk gone")

    import bracket.orchestrator.history as hist
    monkeypatch.setattr(hist.sqlite3, "connect", _boom)
    record_session_winner(
        trainer_name="sdxl", search_space_name="x",
        best_config={"lr": 1e-4}, best_score=0.1, db_path=db,
    )  # must not raise


def test_orchestrator_warmstarts_from_history(tmp_path: Path, monkeypatch):
    """End-to-end: previous session winners get prepended to curated configs
    when use_history_priors=True."""
    from bracket.search.controller import RandomSearch
    from bracket.orchestrator.loop import orchestrate
    from tests.test_orchestrator_loop import MockTrainer, MockConfig

    db = tmp_path / "history.db"
    record_session_winner(
        trainer_name="mock", search_space_name="mock",
        best_config={"target_loss": 0.25},
        best_score=0.1, db_path=db,
    )

    trainer = MockTrainer()
    out = tmp_path / "session"
    orchestrate(
        trainer=trainer,
        dataset_toml=tmp_path / "ds.toml",
        output_dir=out,
        controller=RandomSearch(seed=1),
        budget_runs=2,
        max_steps_per_run=40,
        max_wall_seconds_per_run=60,
        use_history_priors=True,
        history_db_path=db,
    )
    import json
    rows = [
        json.loads(line)
        for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    roles = [r["role"] for r in rows if r["role"] != "setup"]
    # baseline + 1 curated (history) + 1 candidate
    assert "curated" in roles
    cur = next(r for r in rows if r["role"] == "curated")
    assert cur["config"]["target_loss"] == pytest.approx(0.25)


def test_orchestrator_records_session_winner_on_success(tmp_path: Path):
    """After a session completes, the winner must be persisted so the
    next session can warm-start from it."""
    from bracket.search.controller import RandomSearch
    from bracket.orchestrator.loop import orchestrate
    from tests.test_orchestrator_loop import MockTrainer

    db = tmp_path / "history.db"
    trainer = MockTrainer()
    out = tmp_path / "session"
    orchestrate(
        trainer=trainer, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=1), budget_runs=2,
        max_steps_per_run=40, max_wall_seconds_per_run=60,
        history_db_path=db,
    )
    # The mock trainer converges to whatever target_loss the controller
    # picks — at minimum we expect a single row recorded.
    out_configs = lookup_warmstart_configs("mock", "mock", db_path=db)
    assert len(out_configs) >= 1
    assert "target_loss" in out_configs[0]
