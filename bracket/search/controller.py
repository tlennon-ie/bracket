"""Search controllers — pick the next config given history.

v0.1 ships RandomSearch. The interface admits future bandit/Bayesian/LLM-guided
controllers — they just consume the ledger more cleverly.
"""

from __future__ import annotations

import random as _random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from bracket.search.mutator import sample_random
from bracket.search.space import SearchSpace


@dataclass(frozen=True)
class LedgerEntry:
    """One row in the result ledger — what the controller sees as history."""

    run_id: str
    config: Mapping[str, Any]
    score: Optional[float]   # None if the run failed before we could score it
    error: Optional[str]
    duration_s: float
    n_steps: int
    timestamp: float


class SearchController(ABC):
    @abstractmethod
    def next_config(
        self, space: SearchSpace, history: Sequence[LedgerEntry]
    ) -> dict[str, Any]:
        """Pick the next config to try."""

    @abstractmethod
    def should_stop(
        self, history: Sequence[LedgerEntry], *, budget_runs: int
    ) -> bool:
        """Whether to halt before reaching budget_runs (e.g. early-stop on plateau)."""


class RandomSearch(SearchController):
    """Independent uniform sample on each call. Baseline for v0.1."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = _random.Random(seed)

    def next_config(
        self, space: SearchSpace, history: Sequence[LedgerEntry]
    ) -> dict[str, Any]:
        return sample_random(space, self._rng)

    def should_stop(
        self, history: Sequence[LedgerEntry], *, budget_runs: int
    ) -> bool:
        return len(history) >= budget_runs
