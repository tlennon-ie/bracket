"""NSGA-II + Pareto-front tests for the multi-objective controller."""
from __future__ import annotations

import math

import pytest

optuna = pytest.importorskip("optuna")

from bracket.search.optuna_multi import (
    OptunaNSGAIISearch,
    bootstrap_hypervolume_ci,
    compute_pareto_front,
)
from bracket.search.space import FloatKnob, SearchSpace


# ---------------------------------------------------------------------------
# compute_pareto_front
# ---------------------------------------------------------------------------


def test_pareto_recovers_dominated_set_for_known_points() -> None:
    # The Pareto front of f(x) = (x**2, (x-1)**2) for x in [0, 1] is the
    # entire arc; for a discrete sample we recover all points where neither
    # axis strictly dominates the other.
    pts = [(0.0, 1.0), (0.25, 0.5625), (0.5, 0.25), (0.75, 0.0625), (1.0, 0.0)]
    front = compute_pareto_front(pts)
    assert sorted(front) == [0, 1, 2, 3, 4]


def test_pareto_eliminates_dominated_points() -> None:
    # (1, 1) is dominated by (0.5, 0.5); the others form the front.
    pts = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0), (1.0, 1.0)]
    front = compute_pareto_front(pts)
    assert sorted(front) == [0, 1, 2]


def test_pareto_handles_nan_points() -> None:
    pts = [(0.0, 1.0), (float("nan"), 0.1), (1.0, 0.0), (0.5, 0.5)]
    front = compute_pareto_front(pts)
    # NaN point (idx 1) is excluded; remaining front: 0, 2, 3.
    assert sorted(front) == [0, 2, 3]


def test_pareto_empty_input() -> None:
    assert compute_pareto_front([]) == []


# ---------------------------------------------------------------------------
# OptunaNSGAIISearch
# ---------------------------------------------------------------------------


def _space() -> SearchSpace:
    return SearchSpace(
        name="t",
        knobs={"x": FloatKnob(low=0.0, high=1.0)},
    )


def test_nsga2_samples_within_space() -> None:
    c = OptunaNSGAIISearch(seed=42)
    space = _space()
    for _ in range(10):
        s = c.next_config(space, [])
        assert 0.0 <= s["x"] <= 1.0


def test_nsga2_rejects_wrong_objective_count() -> None:
    with pytest.raises(ValueError):
        OptunaNSGAIISearch(objectives=("only_one",))
    with pytest.raises(ValueError):
        OptunaNSGAIISearch(objectives=("a", "b", "c"))


def test_nsga2_rejects_duplicate_objectives() -> None:
    with pytest.raises(ValueError):
        OptunaNSGAIISearch(objectives=("score", "score"))


def test_nsga2_recovers_pareto_front_for_synthetic_objective() -> None:
    """For f(x) = (x**2, (x-1)**2) on x in [0, 1], the entire interval is
    Pareto-optimal. After ~16 trials, the best_trials list should be
    non-empty and every reported pair should lie on (approximately) the
    optimal arc."""
    from bracket.orchestrator.loop import config_id

    c = OptunaNSGAIISearch(seed=0, population_size=4)
    space = _space()
    for _ in range(16):
        sample = c.next_config(space, [])
        x = sample["x"]
        c.observe_multi(config_id(sample), (x * x, (x - 1) * (x - 1)))
    front = c.pareto_front()
    assert len(front) >= 1
    for _id, (a, b), params in front:
        x = params["x"]
        # All front points must be feasible (x in [0,1]).
        assert 0.0 <= x <= 1.0
        # And must lie on the convex arc: a + b in [0.5, 1.0] for x in [0,1].
        # Strict tolerance: a = x^2, b = (1-x)^2 → a + b ranges 0.5 to 1.0.
        assert 0.49 <= a + b <= 1.01


def test_nsga2_observe_multi_handles_none_values() -> None:
    """A disqualified trial reports None for one or both objectives. The
    controller must accept these without crashing — it converts them to
    a large finite value internally."""
    from bracket.orchestrator.loop import config_id

    c = OptunaNSGAIISearch(seed=0)
    space = _space()
    sample = c.next_config(space, [])
    # observe_multi accepts None on either axis; should not raise.
    c.observe_multi(config_id(sample), (None, 0.5))  # type: ignore[arg-type]
    # And another with both None.
    sample2 = c.next_config(space, [])
    c.observe_multi(config_id(sample2), (None, None))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def test_bootstrap_hypervolume_returns_finite_band() -> None:
    points = [(0.1, 0.9), (0.3, 0.4), (0.6, 0.2), (0.8, 0.1)]
    median, lo, hi = bootstrap_hypervolume_ci(
        points, reference=(1.0, 1.0), n_resamples=200, seed=0,
    )
    assert lo <= median <= hi
    assert median > 0.0


def test_bootstrap_hypervolume_empty() -> None:
    median, lo, hi = bootstrap_hypervolume_ci([], reference=(1.0, 1.0))
    assert (median, lo, hi) == (0.0, 0.0, 0.0)
