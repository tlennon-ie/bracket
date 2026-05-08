"""Exponential moving average for the noisy per-step diffusion loss.

The user's TensorBoard chart shows loss/current bouncing between ~0.30 and
~0.37 — that variance comes from sampling different diffusion timesteps each
step, not from the optimizer. EMA pulls a readable trend out of it.

Two recommended alpha pairings:
  - fast (alpha=0.1)  → lags ~10 steps; tracks meaningful changes
  - slow (alpha=0.01) → lags ~100 steps; the "is it actually converging" view
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EMASmoother:
    """Single-alpha EMA. New value = alpha*x + (1-alpha)*previous.

    First update seeds the state with the raw value rather than starting from
    zero, so the smoothed line doesn't ramp up from 0.0 over the first ~50 steps.
    """

    alpha: float
    _value: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = float(x)
        else:
            self._value = self.alpha * float(x) + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        self._value = None
