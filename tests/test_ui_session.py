"""Tests for the UI's orchestration session wrapper. We don't actually start
Gradio — just verify the threading + state-snapshot semantics."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from bracket.orchestrator.loop import OrchestrationResult
from bracket.search.controller import LedgerEntry
from bracket.ui.session import OrchestrationSession


def _fake_result(out: Path) -> OrchestrationResult:
    return OrchestrationResult(
        session_dir=out, ledger_path=out / "ledger.jsonl",
        history=[], baseline_entry=None, best_entry=None,
        n_completed=0, n_disqualified=0,
    )


def test_session_starts_and_finishes(tmp_path: Path):
    sess = OrchestrationSession()
    finished = []

    def run_fn() -> OrchestrationResult:
        logging.getLogger("bracket").info("hello world")
        time.sleep(0.05)
        finished.append(True)
        return _fake_result(tmp_path)

    sess.start(run_fn, output_dir=tmp_path, total_runs_target=2)
    # Wait
    deadline = time.time() + 5.0
    while sess.is_running() and time.time() < deadline:
        time.sleep(0.05)
    snap = sess.snapshot()
    assert snap.status == "done"
    assert finished == [True]
    assert snap.error_message is None


def test_session_captures_log_lines(tmp_path: Path):
    sess = OrchestrationSession()

    def run_fn() -> OrchestrationResult:
        log = logging.getLogger("bracket")
        log.info("running baseline baseline-000-1")
        log.info("baseline score=0.3 dq=None")
        log.info("running candidate 1/1 cand-000-1 knobs={}")
        log.info("candidate cand-000-1 score=0.2 dq=None duration=1.0s")
        return _fake_result(tmp_path)

    sess.start(run_fn, output_dir=tmp_path, total_runs_target=2)
    deadline = time.time() + 5.0
    while sess.is_running() and time.time() < deadline:
        time.sleep(0.05)
    snap = sess.snapshot()
    assert snap.status == "done"
    # Heuristic counter: "score=" lines for baseline + candidate
    assert snap.completed_runs >= 2
    assert any("score=0.2" in line for line in snap.log_tail)


def test_session_records_error_message(tmp_path: Path):
    sess = OrchestrationSession()

    def run_fn() -> OrchestrationResult:
        raise RuntimeError("simulated failure")

    sess.start(run_fn, output_dir=tmp_path, total_runs_target=1)
    deadline = time.time() + 5.0
    while sess.is_running() and time.time() < deadline:
        time.sleep(0.05)
    snap = sess.snapshot()
    assert snap.status == "error"
    assert "simulated failure" in (snap.error_message or "")


def test_session_rejects_concurrent_starts(tmp_path: Path):
    sess = OrchestrationSession()

    def run_fn() -> OrchestrationResult:
        time.sleep(0.5)
        return _fake_result(tmp_path)

    sess.start(run_fn, output_dir=tmp_path, total_runs_target=1)
    try:
        sess.start(run_fn, output_dir=tmp_path, total_runs_target=1)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    # Wait for it to complete so the test cleans up
    deadline = time.time() + 5.0
    while sess.is_running() and time.time() < deadline:
        time.sleep(0.05)
