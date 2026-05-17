"""Shared pytest fixtures.

The orchestrator unit tests pass synthetic ``tmp_path / "ds.toml"`` paths
to mock trainers — those TOMLs are never written to disk, so the real
``validate_dataset_toml`` check at the top of ``orchestrate()`` would
fail with "Dataset TOML does not exist" before any orchestration logic
runs. We side-step that with an autouse fixture that swaps in a
permissive stub for these modules only.

Tests in ``test_dataset_validator.py`` keep using the real validator
because they exercise it directly with on-disk fixtures.
"""
from __future__ import annotations

from typing import Iterator

import pytest

from bracket.dataset.validator import ValidationResult
from bracket.orchestrator import loop as _orchestrator_loop


@pytest.fixture(autouse=True)
def _isolate_history_db(tmp_path_factory, monkeypatch) -> Iterator[None]:
    """Point ``BRACKET_HISTORY_DB`` at a throwaway path so orchestrator
    tests that don't explicitly pass ``history_db_path`` never touch the
    user's real ``~/.cache/bracket/history.db``."""
    db = tmp_path_factory.mktemp("history") / "history.db"
    monkeypatch.setenv("BRACKET_HISTORY_DB", str(db))
    yield


_PASS = ValidationResult(is_valid=True, errors=[], summary="ok (test stub)")


@pytest.fixture(autouse=True)
def _stub_dataset_validation_for_orchestrator_tests(
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Replace ``validate_dataset_toml`` with a no-op for tests that
    drive ``orchestrate()`` through a MockTrainer. The fixture is
    autouse but module-scoped: it only swaps the stub in for the
    orchestrator/finals/resume test files."""
    # pytest reports the module as ``tests.test_<name>`` once
    # tests/__init__.py is present (it is). Strip any prefix before
    # comparing so the stub still applies if the layout changes.
    module_name = request.node.module.__name__.rsplit(".", 1)[-1]
    targeted = {
        "test_orchestrator_loop",
        "test_resume_guard",
        "test_finals",
        "test_history",
    }
    if module_name not in targeted:
        yield
        return
    original = _orchestrator_loop.validate_dataset_toml
    _orchestrator_loop.validate_dataset_toml = lambda _path: _PASS  # type: ignore[assignment]
    try:
        yield
    finally:
        _orchestrator_loop.validate_dataset_toml = original  # type: ignore[assignment]
