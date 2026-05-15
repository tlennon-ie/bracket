"""Declarative search-space primitives.

A SearchSpace is a name → Knob mapping. Each Knob knows how to sample a value
and how to validate one. Float knobs support log-uniform sampling because lr
ranges span orders of magnitude.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


class Knob:
    """Base class. Subclasses must implement sample(rng) and validate(value)."""

    def sample(self, rng: _random.Random) -> Any:
        raise NotImplementedError

    def validate(self, value: Any) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class FloatKnob(Knob):
    low: float
    high: float
    log: bool = False  # log-uniform sampling (use for lr, weight decay, etc.)

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high {self.high} < low {self.low}")
        if self.log and self.low <= 0:
            raise ValueError("log=True requires low > 0")

    def sample(self, rng: _random.Random) -> float:
        if self.log:
            return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        return rng.uniform(self.low, self.high)

    def validate(self, value: Any) -> None:
        v = float(value)
        if not (self.low <= v <= self.high):
            raise ValueError(f"FloatKnob value {v} outside [{self.low}, {self.high}]")


@dataclass(frozen=True)
class IntKnob(Knob):
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high {self.high} < low {self.low}")

    def sample(self, rng: _random.Random) -> int:
        return rng.randint(self.low, self.high)

    def validate(self, value: Any) -> None:
        v = int(value)
        if not (self.low <= v <= self.high):
            raise ValueError(f"IntKnob value {v} outside [{self.low}, {self.high}]")


@dataclass(frozen=True)
class CategoricalKnob(Knob):
    choices: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("CategoricalKnob requires at least one choice")

    def sample(self, rng: _random.Random) -> Any:
        return rng.choice(list(self.choices))

    def validate(self, value: Any) -> None:
        if value not in self.choices:
            raise ValueError(f"CategoricalKnob value {value!r} not in {self.choices}")


@dataclass(frozen=True)
class FixedKnob(Knob):
    """A "knob" that doesn't actually vary — useful for documenting choices
    that are pinned for this search and shipping the same key shape as the
    variable knobs."""

    value: Any

    def sample(self, rng: _random.Random) -> Any:  # noqa: ARG002
        return self.value

    def validate(self, value: Any) -> None:
        if value != self.value:
            raise ValueError(f"FixedKnob expected {self.value!r}, got {value!r}")


@dataclass(frozen=True)
class SearchSpace:
    """A bag of named knobs, plus an optional human-readable name."""

    name: str
    knobs: Mapping[str, Knob]

    def sample(self, rng: _random.Random) -> dict[str, Any]:
        return {name: knob.sample(rng) for name, knob in self.knobs.items()}

    def validate(self, sample: Mapping[str, Any]) -> None:
        missing = set(self.knobs) - set(sample)
        extra = set(sample) - set(self.knobs)
        if missing:
            raise ValueError(f"sample missing knobs: {sorted(missing)}")
        if extra:
            raise ValueError(f"sample has unknown knobs: {sorted(extra)}")
        for name, knob in self.knobs.items():
            knob.validate(sample[name])


@dataclass(frozen=True)
class SearchOverrides:
    """User-supplied narrowing for the three knobs we expose in the UI.

    ``None`` on any field = leave the trainer's declared knob alone.
    Apply via :func:`apply_search_overrides`. Keys that don't exist on the
    target trainer's search space are silently ignored — every diffusion
    trainer in the registry today has these three knob names, but the
    contract is "best-effort narrow" so future trainers don't have to.
    """

    lr_min: Optional[float] = None
    lr_max: Optional[float] = None
    batch_size_min: Optional[int] = None
    batch_size_max: Optional[int] = None
    # "on" → FixedKnob(True); "off" → FixedKnob(False); "search" →
    # CategoricalKnob((True, False)). Anything else (including None) is
    # a no-op so the trainer's tier-aware default wins.
    gradient_checkpointing_mode: Optional[str] = None

    def is_empty(self) -> bool:
        return (
            self.lr_min is None
            and self.lr_max is None
            and self.batch_size_min is None
            and self.batch_size_max is None
            and self.gradient_checkpointing_mode in (None, "")
        )


def apply_search_overrides(space: SearchSpace, overrides: Optional[SearchOverrides]) -> SearchSpace:
    """Return a new SearchSpace with the user-supplied bounds applied.

    Behaviour per knob:

    * ``learning_rate`` (``FloatKnob``): rebuilds with ``lr_min`` / ``lr_max``;
      either bound may be omitted and the trainer's existing value is kept.
    * ``train_batch_size`` (``CategoricalKnob`` of ints, or ``IntKnob`` /
      ``FixedKnob``): if categorical, filters the choice list by the
      ``[batch_size_min, batch_size_max]`` window; if no existing choice
      survives, falls back to an ``IntKnob`` so search still produces a
      valid sample.
    * ``gradient_checkpointing``: replaces the knob entirely with a
      ``FixedKnob``/``CategoricalKnob`` per the ``mode``.

    The returned space's ``name`` gets a ``+overrides`` suffix so the
    resume guard in :func:`bracket.orchestrator.loop.orchestrate` refuses
    to mix runs with different search bounds in one ``output_dir``.
    """

    if overrides is None or overrides.is_empty():
        return space

    new_knobs: dict[str, Knob] = dict(space.knobs)

    if (overrides.lr_min is not None or overrides.lr_max is not None) \
            and "learning_rate" in new_knobs:
        current = new_knobs["learning_rate"]
        low = overrides.lr_min if overrides.lr_min is not None else getattr(current, "low", 1e-6)
        high = overrides.lr_max if overrides.lr_max is not None else getattr(current, "high", 5e-4)
        low_f, high_f = float(low), float(high)
        if low_f > high_f:
            low_f, high_f = high_f, low_f
        new_knobs["learning_rate"] = FloatKnob(low=low_f, high=high_f, log=True)

    if (overrides.batch_size_min is not None or overrides.batch_size_max is not None) \
            and "train_batch_size" in new_knobs:
        current = new_knobs["train_batch_size"]
        bs_min = int(overrides.batch_size_min) if overrides.batch_size_min is not None else 1
        bs_max = int(overrides.batch_size_max) if overrides.batch_size_max is not None else max(bs_min, 8)
        if bs_min > bs_max:
            bs_min, bs_max = bs_max, bs_min
        if isinstance(current, CategoricalKnob):
            try:
                filtered = tuple(c for c in current.choices if bs_min <= int(c) <= bs_max)
            except (TypeError, ValueError):
                filtered = tuple()
            if filtered:
                new_knobs["train_batch_size"] = CategoricalKnob(choices=filtered)
            else:
                new_knobs["train_batch_size"] = IntKnob(low=bs_min, high=bs_max)
        else:
            new_knobs["train_batch_size"] = IntKnob(low=bs_min, high=bs_max)

    gc_mode = overrides.gradient_checkpointing_mode
    if gc_mode and "gradient_checkpointing" in new_knobs:
        if gc_mode == "on":
            new_knobs["gradient_checkpointing"] = FixedKnob(value=True)
        elif gc_mode == "off":
            new_knobs["gradient_checkpointing"] = FixedKnob(value=False)
        elif gc_mode == "search":
            new_knobs["gradient_checkpointing"] = CategoricalKnob(choices=(True, False))

    return SearchSpace(name=f"{space.name}+overrides", knobs=new_knobs)
