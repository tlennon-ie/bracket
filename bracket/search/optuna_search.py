"""Optuna TPE search controller.

Drop-in replacement for RandomSearch that uses Tree-structured Parzen
Estimator (TPE) sampling — learns from each completed candidate what's
promising and biases the next sample toward those regions.

Compatible with the orchestrator's resume semantics: history from prior
sessions is replayed into a fresh Study at construction time, so TPE has
all observed (config → score) pairs to learn from.

When optuna isn't installed, the import-time guard raises a friendly
ImportError; the rest of the package keeps working without optuna.
"""

from __future__ import annotations

from typing import Any, Sequence

from bracket.search.controller import LedgerEntry, SearchController
from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchSpace,
)


try:
    import optuna  # noqa: F401  (presence-check)
    from optuna.samplers import TPESampler
    from optuna.trial import Trial
    _OPTUNA_OK = True
except ImportError:  # pragma: no cover
    _OPTUNA_OK = False


def _require_optuna() -> None:
    if not _OPTUNA_OK:
        raise ImportError(
            "optuna is required for OptunaTPESearch. "
            "pip install optuna"
        )


def _suggest(trial: "Trial", name: str, knob) -> Any:
    if isinstance(knob, FixedKnob):
        return knob.value
    if isinstance(knob, FloatKnob):
        return trial.suggest_float(name, knob.low, knob.high, log=knob.log)
    if isinstance(knob, IntKnob):
        return trial.suggest_int(name, knob.low, knob.high)
    if isinstance(knob, CategoricalKnob):
        # Optuna requires hashable choices — Float values are fine.
        return trial.suggest_categorical(name, list(knob.choices))
    raise TypeError(f"unsupported knob type {type(knob).__name__}")


class OptunaTPESearch(SearchController):
    """TPE-guided sampler. Same interface as RandomSearch."""

    def __init__(
        self,
        seed: int = 0,
        *,
        n_startup_trials: int = 5,
        multivariate: bool = True,
    ) -> None:
        _require_optuna()
        self._seed = seed
        self._n_startup = n_startup_trials
        self._multivariate = multivariate
        self._study: "optuna.Study" | None = None
        self._space: SearchSpace | None = None
        self._replayed_run_ids: set[str] = set()
        self._pending_trials: dict[str, "Trial"] = {}

    def _ensure_study(self, space: SearchSpace) -> None:
        if self._study is not None and self._space is space:
            return
        sampler = TPESampler(
            seed=self._seed,
            n_startup_trials=self._n_startup,
            multivariate=self._multivariate,
            # Silence the per-trial "dynamic search space" warning. With
            # multivariate=True, TPE falls back to independent per-param
            # sampling on trials it can't fit jointly (e.g. when historical
            # disqualifications create score-range outliers). The fallback
            # is desired behaviour; the warning just spams the console.
            warn_independent_sampling=False,
        )
        self._study = optuna.create_study(direction="minimize", sampler=sampler)
        self._space = space
        self._replayed_run_ids.clear()
        self._pending_trials.clear()

    def _replay_history(self, history: Sequence[LedgerEntry]) -> None:
        """Tell Optuna about prior runs — both successes and disqualifications.

        Disqualifications are added as PRUNED trials with a high-but-finite
        score; failures are logged with infinity. TPE handles pruned trials
        by considering them poor samples without polluting the score
        distribution.
        """
        assert self._study is not None
        for h in history:
            if h.run_id in self._replayed_run_ids:
                continue
            self._replayed_run_ids.add(h.run_id)
            params = {k: v for k, v in dict(h.config).items() if k in self._space.knobs}
            distributions = self._space_to_distributions()
            # filter to keys that actually exist in distributions (skip unrecognised legacy keys)
            params = {k: v for k, v in params.items() if k in distributions}
            distributions = {k: distributions[k] for k in params}
            if h.score is None:
                # Disqualified or failed — register as a finished trial with
                # large value so TPE learns to avoid this region.
                value = 1e6
            else:
                value = float(h.score)
            try:
                trial = optuna.trial.create_trial(
                    params=params, distributions=distributions, value=value,
                )
                self._study.add_trial(trial)
            except Exception:
                # Best-effort; don't crash the controller on a malformed historical row.
                continue

    def _space_to_distributions(self) -> dict[str, "optuna.distributions.BaseDistribution"]:
        assert self._space is not None
        from optuna import distributions
        out: dict[str, distributions.BaseDistribution] = {}
        for name, knob in self._space.knobs.items():
            if isinstance(knob, FixedKnob):
                out[name] = distributions.CategoricalDistribution(choices=[knob.value])
            elif isinstance(knob, FloatKnob):
                out[name] = distributions.FloatDistribution(
                    low=knob.low, high=knob.high, log=knob.log,
                )
            elif isinstance(knob, IntKnob):
                out[name] = distributions.IntDistribution(low=knob.low, high=knob.high)
            elif isinstance(knob, CategoricalKnob):
                out[name] = distributions.CategoricalDistribution(choices=list(knob.choices))
        return out

    def next_config(
        self, space: SearchSpace, history: Sequence[LedgerEntry]
    ) -> dict[str, Any]:
        self._ensure_study(space)
        self._replay_history(history)
        assert self._study is not None
        trial = self._study.ask()
        sample: dict[str, Any] = {}
        for name, knob in space.knobs.items():
            sample[name] = _suggest(trial, name, knob)
        # Stash the trial keyed by a tuple of (sorted items) so we can tell()
        # back when a future call observes the score for this config. We use
        # the canonical hash from loop.config_id at observation time.
        from bracket.orchestrator.loop import config_id as _cid
        cid = _cid(sample)
        self._pending_trials[cid] = trial
        return sample

    def should_stop(
        self, history: Sequence[LedgerEntry], *, budget_runs: int
    ) -> bool:
        # Match RandomSearch semantics: count of unique candidate configs.
        from bracket.orchestrator.loop import config_id as _cid
        unique_cids = {_cid(dict(h.config)) for h in history if not h.run_id.startswith("baseline-")}
        return len(unique_cids) >= budget_runs

    def observe(self, config_id: str, score: float | None) -> None:
        """Optional: feed a freshly-completed config's score back into the
        study so TPE updates its model. The orchestrator can call this after
        each candidate; alternatively the next next_config() call replays the
        ledger and picks up everything anyway."""
        if not _OPTUNA_OK or self._study is None:
            return
        trial = self._pending_trials.pop(config_id, None)
        if trial is None:
            return
        try:
            self._study.tell(trial, value=float(score) if score is not None else 1e6)
        except Exception:
            pass
