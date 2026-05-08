"""Declarative search-space primitives.

A SearchSpace is a name → Knob mapping. Each Knob knows how to sample a value
and how to validate one. Float knobs support log-uniform sampling because lr
ranges span orders of magnitude.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Any, Mapping


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
