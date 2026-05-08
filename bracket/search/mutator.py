"""Sample / perturb operators over a SearchSpace.

`sample_random` is for v0.1 random search. `perturb` is wired in for future
evolutionary or local-search variants — it tweaks a known-good config rather
than sampling from scratch.
"""

from __future__ import annotations

import math
import random as _random
from typing import Any, Mapping

from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchSpace,
)


def sample_random(space: SearchSpace, rng: _random.Random) -> dict[str, Any]:
    """One independent sample over each knob — the standard random search step."""
    return space.sample(rng)


def perturb(
    base: Mapping[str, Any],
    space: SearchSpace,
    rng: _random.Random,
    *,
    sigma: float = 0.25,
    fraction_changed: float = 0.5,
) -> dict[str, Any]:
    """Mutate a fraction of knobs in `base` to nearby values.

    For float knobs: multiply by exp(N(0, sigma)) when log, else add sigma*range*N(0,1).
    For int knobs: jitter by floor(sigma * range * N(0,1)), clipped to [low, high].
    For categorical: with probability sigma, resample.
    For fixed: never changes.
    """
    space.validate(base)
    changed_keys = list(space.knobs)
    rng.shuffle(changed_keys)
    n_change = max(1, int(round(fraction_changed * len(changed_keys))))
    to_change = set(changed_keys[:n_change])

    out: dict[str, Any] = dict(base)
    for name, knob in space.knobs.items():
        if name not in to_change:
            continue
        if isinstance(knob, FixedKnob):
            continue
        cur = base[name]
        if isinstance(knob, FloatKnob):
            if knob.log:
                noisy = float(cur) * math.exp(rng.gauss(0.0, sigma))
            else:
                width = knob.high - knob.low
                noisy = float(cur) + rng.gauss(0.0, sigma * width)
            out[name] = max(knob.low, min(knob.high, noisy))
        elif isinstance(knob, IntKnob):
            width = knob.high - knob.low
            jitter = int(round(rng.gauss(0.0, sigma * max(1, width))))
            out[name] = max(knob.low, min(knob.high, int(cur) + jitter))
        elif isinstance(knob, CategoricalKnob):
            if rng.random() < sigma:
                out[name] = knob.sample(rng)
        else:
            raise TypeError(f"unknown knob type {type(knob).__name__}")
    space.validate(out)
    return out
