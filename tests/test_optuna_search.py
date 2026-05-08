"""Optuna TPE controller tests — verify it samples within the search space,
respects the SearchController interface, and stops at the configured budget."""
from __future__ import annotations

import pytest

optuna = pytest.importorskip("optuna")

from bracket.search.controller import LedgerEntry
from bracket.search.optuna_search import OptunaTPESearch
from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchSpace,
)


def _space() -> SearchSpace:
    return SearchSpace(
        name="t",
        knobs={
            "lr": FloatKnob(low=1e-6, high=1e-3, log=True),
            "warmup": IntKnob(low=0, high=200),
            "opt": CategoricalKnob(choices=("AdamW", "Lion", "Prodigy")),
            "rank": FixedKnob(value=32),
        },
    )


def test_optuna_samples_validate_against_space():
    space = _space()
    c = OptunaTPESearch(seed=0)
    for _ in range(8):
        s = c.next_config(space, [])
        space.validate(s)


def test_optuna_log_float_stays_in_range():
    space = _space()
    c = OptunaTPESearch(seed=42)
    for _ in range(20):
        s = c.next_config(space, [])
        assert 1e-6 <= s["lr"] <= 1e-3
        assert 0 <= s["warmup"] <= 200
        assert s["opt"] in ("AdamW", "Lion", "Prodigy")
        assert s["rank"] == 32


def test_optuna_should_stop_counts_unique_configs():
    space = _space()
    c = OptunaTPESearch(seed=1)
    history: list[LedgerEntry] = []
    assert not c.should_stop(history, budget_runs=3)
    # Two seeds of the same config = 1 unique
    cfg = {"lr": 1e-4, "warmup": 50, "opt": "AdamW", "rank": 32}
    history.append(LedgerEntry(
        run_id="cand-000-s0", config=cfg, score=0.3, error=None,
        duration_s=1.0, n_steps=10, timestamp=0.0,
    ))
    history.append(LedgerEntry(
        run_id="cand-000-s1", config=cfg, score=0.31, error=None,
        duration_s=1.0, n_steps=10, timestamp=0.0,
    ))
    assert not c.should_stop(history, budget_runs=3)
    history.append(LedgerEntry(
        run_id="cand-001-s0",
        config={"lr": 5e-5, "warmup": 30, "opt": "Lion", "rank": 32},
        score=0.25, error=None, duration_s=1.0, n_steps=10, timestamp=0.0,
    ))
    history.append(LedgerEntry(
        run_id="cand-002-s0",
        config={"lr": 2e-4, "warmup": 100, "opt": "Prodigy", "rank": 32},
        score=0.20, error=None, duration_s=1.0, n_steps=10, timestamp=0.0,
    ))
    assert c.should_stop(history, budget_runs=3)


def test_optuna_replays_history_without_crashing():
    """Replaying historical observations into a fresh study must succeed.
    Stronger "TPE actually concentrates near the minimum" testing is flaky —
    we verify the integration here, not the algorithm's emergent behaviour."""
    space = SearchSpace(
        name="t",
        knobs={"lr": FloatKnob(low=1e-6, high=1e-3, log=True)},
    )
    history = [
        LedgerEntry(
            run_id=f"cand-{i:03d}-s0",
            config={"lr": 2e-4 + 0.05 * (i - 5) * 1e-5},
            score=0.05 + abs(i - 5) * 0.05,
            error=None, duration_s=1.0, n_steps=10, timestamp=float(i),
        )
        for i in range(20)
    ]
    c = OptunaTPESearch(seed=0, n_startup_trials=0)
    for _ in range(5):
        s = c.next_config(space, history)
        space.validate(s)


def test_optuna_handles_disqualified_history_gracefully():
    space = _space()
    c = OptunaTPESearch(seed=0)
    history = [LedgerEntry(
        run_id="cand-000-s0",
        config={"lr": 1e-4, "warmup": 50, "opt": "Lion", "rank": 32},
        score=None,  # disqualified
        error="diverging", duration_s=1.0, n_steps=10, timestamp=0.0,
    )]
    s = c.next_config(space, history)
    space.validate(s)
