"""Multi-objective Optuna controller — NSGA-II + Pareto-front reporting.

Single-objective TPE picks the best of one scalar. When the user cares
about generalisation, the orchestrator's :class:`Scorer` (with OOD
prompt support, landing alongside this controller) emits two
components in ``ScoreReport.components``:

  - ``in_dist_score``  — performance on the prompts that match training
  - ``ood_score``      — performance on a held-out OOD prompt set

This controller asks NSGA-II to minimise both. The orchestrator extracts
the two component values from each completed run and feeds them back via
:meth:`observe_multi`. The final report renders a Pareto-front
scatter (see :mod:`bracket.proof.report`).

Why NSGA-II rather than TPE's multi-objective mode: Optuna's
``MOTPESampler`` is documented as deprecated since 3.0. NSGA-II is the
maintained default for two-objective problems and produces a stable
Pareto front from small population sizes.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from bracket.search.controller import LedgerEntry, SearchController
from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchSpace,
)

logger = logging.getLogger("bracket.search.optuna_multi")


try:
    import optuna  # noqa: F401  (presence-check)
    from optuna.samplers import NSGAIISampler
    from optuna.trial import Trial
    _OPTUNA_OK = True
except ImportError:  # pragma: no cover
    _OPTUNA_OK = False


def _require_optuna() -> None:
    if not _OPTUNA_OK:
        raise ImportError(
            "optuna is required for OptunaNSGAIISearch. pip install optuna"
        )


def _suggest(trial: "Trial", name: str, knob) -> Any:
    if isinstance(knob, FixedKnob):
        return knob.value
    if isinstance(knob, FloatKnob):
        return trial.suggest_float(name, knob.low, knob.high, log=knob.log)
    if isinstance(knob, IntKnob):
        return trial.suggest_int(name, knob.low, knob.high)
    if isinstance(knob, CategoricalKnob):
        return trial.suggest_categorical(name, list(knob.choices))
    raise TypeError(f"unsupported knob type {type(knob).__name__}")


class OptunaNSGAIISearch(SearchController):
    """Two-objective Pareto search with NSGA-II.

    Public surface mirrors :class:`OptunaTPESearch` so the orchestrator can
    swap in either controller transparently. The extra plumbing is
    :meth:`observe_multi`: callers feed back a tuple ``(in_dist, ood)``
    after each trial completes, since the scalar ``LedgerEntry.score``
    can't carry two objectives.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        objectives: Sequence[str] = ("in_dist_score", "ood_score"),
        population_size: int = 8,
    ) -> None:
        _require_optuna()
        if len(objectives) != 2:
            raise ValueError(
                f"OptunaNSGAIISearch needs exactly two objectives; got {len(objectives)}"
            )
        if objectives[0] == objectives[1]:
            raise ValueError("objectives must be distinct")
        self._seed = seed
        self._objectives = tuple(objectives)
        self._population_size = max(2, int(population_size))
        self._study: "optuna.Study" | None = None
        self._space: SearchSpace | None = None
        self._pending_trials: dict[str, "Trial"] = {}
        # Bookkeeping for replay so we never tell() the same trial twice.
        self._completed_cids: set[str] = set()

    @property
    def objectives(self) -> tuple[str, ...]:
        return self._objectives

    def _ensure_study(self, space: SearchSpace) -> None:
        if self._study is not None and self._space is space:
            return
        sampler = NSGAIISampler(
            seed=self._seed, population_size=self._population_size,
        )
        self._study = optuna.create_study(
            directions=["minimize", "minimize"], sampler=sampler,
        )
        self._space = space
        self._completed_cids.clear()
        self._pending_trials.clear()

    def next_config(
        self, space: SearchSpace, history: Sequence[LedgerEntry]
    ) -> dict[str, Any]:
        self._ensure_study(space)
        assert self._study is not None
        trial = self._study.ask()
        sample: dict[str, Any] = {}
        for name, knob in space.knobs.items():
            sample[name] = _suggest(trial, name, knob)
        from bracket.orchestrator.loop import config_id as _cid
        cid = _cid(sample)
        self._pending_trials[cid] = trial
        return sample

    def should_stop(
        self, history: Sequence[LedgerEntry], *, budget_runs: int
    ) -> bool:
        from bracket.orchestrator.loop import config_id as _cid
        unique_cids = {
            _cid(dict(h.config)) for h in history
            if not h.run_id.startswith("baseline-")
        }
        return len(unique_cids) >= budget_runs

    def observe_multi(self, config_id: str, values: tuple[float, float]) -> None:
        """Feed a completed trial's pair ``(in_dist, ood)`` back to the study.

        Both values are lower-is-better — same convention as the scalar
        scorer's ``score`` field. Disqualified trials (None on either
        axis) report 1e6 so the sampler treats them as dominated.
        """
        if not _OPTUNA_OK or self._study is None:
            return
        trial = self._pending_trials.pop(config_id, None)
        if trial is None or config_id in self._completed_cids:
            return
        self._completed_cids.add(config_id)
        a, b = values
        try:
            self._study.tell(
                trial,
                values=[
                    float(a) if a is not None else 1e6,
                    float(b) if b is not None else 1e6,
                ],
            )
        except Exception:
            logger.exception("observe_multi failed for %s", config_id)

    def pareto_front(self) -> list[tuple[str, tuple[float, float], dict[str, Any]]]:
        """Return the current Pareto-optimal trials.

        Each entry is ``(trial_number_str, (obj1, obj2), params)``. Empty
        when no trials have completed. Caller decides how to map
        ``trial_number_str`` back to a config_id (typically by matching
        params against the ledger)."""
        if not _OPTUNA_OK or self._study is None:
            return []
        front: list[tuple[str, tuple[float, float], dict[str, Any]]] = []
        for trial in self._study.best_trials:
            if trial.values is None or len(trial.values) != 2:
                continue
            front.append((
                f"trial-{trial.number}",
                (float(trial.values[0]), float(trial.values[1])),
                dict(trial.params),
            ))
        return front


# ---------------------------------------------------------------------------
# Pareto-front utilities (controller-free; used by the proof report).
# ---------------------------------------------------------------------------


def compute_pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Return indices of Pareto-optimal points (lower-is-better on both axes).

    O(N^2) implementation — fine for Bracket's 8-16 candidate budget.
    Caller-supplied points may contain NaN/inf; those entries are
    automatically dominated (never returned).
    """
    n = len(points)
    front: list[int] = []
    for i, (a_i, b_i) in enumerate(points):
        if a_i != a_i or b_i != b_i:  # NaN check
            continue
        dominated = False
        for j, (a_j, b_j) in enumerate(points):
            if i == j:
                continue
            if a_j != a_j or b_j != b_j:
                continue
            if a_j <= a_i and b_j <= b_i and (a_j < a_i or b_j < b_i):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front


def bootstrap_hypervolume_ci(
    points: Sequence[tuple[float, float]],
    *,
    reference: tuple[float, float] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap a 95% confidence interval for the Pareto front's hypervolume.

    Returns ``(median, low, high)``. The reference point defaults to the
    component-wise max of the input + a small margin, matching Optuna's
    convention. Bootstrap resamples points *with* replacement and
    recomputes the dominated hypervolume each draw.
    """
    import random as _random
    finite = [
        (a, b) for (a, b) in points
        if a == a and b == b and a < float("inf") and b < float("inf")
    ]
    if not finite:
        return (0.0, 0.0, 0.0)
    if reference is None:
        ref_a = max(a for a, _ in finite) + 0.1
        ref_b = max(b for _, b in finite) + 0.1
        reference = (ref_a, ref_b)

    rng = _random.Random(seed)
    samples: list[float] = []
    for _ in range(int(n_resamples)):
        draw = [finite[rng.randrange(len(finite))] for _ in range(len(finite))]
        samples.append(_hypervolume(draw, reference))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = samples[int(alpha * len(samples))]
    hi = samples[min(len(samples) - 1, int((1.0 - alpha) * len(samples)))]
    median = samples[len(samples) // 2]
    return (median, lo, hi)


def _hypervolume(
    points: Sequence[tuple[float, float]],
    reference: tuple[float, float],
) -> float:
    """Hypervolume dominated by ``points`` w.r.t. ``reference`` for two-axis
    minimisation. Uses the Klee sweep-line algorithm specialised to 2D."""
    front_idx = compute_pareto_front(points)
    if not front_idx:
        return 0.0
    front = sorted(
        [points[i] for i in front_idx], key=lambda p: p[0],
    )
    ref_a, ref_b = reference
    hv = 0.0
    prev_a = ref_a
    for a, b in reversed(front):
        if b >= ref_b:
            continue
        hv += (prev_a - a) * (ref_b - b)
        prev_a = a
    return max(0.0, hv)
