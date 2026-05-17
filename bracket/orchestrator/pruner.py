"""Early-kill / promotion gate for the orchestration loop.

Two strategies, kept deliberately separate so they can be enabled
independently from :class:`StartSessionRequest`:

  * :class:`DivergenceKiller` — kills a running trial whose loss diverges
    obviously vs. completed peers. Cheap; default ON. NaN/inf hard-kills
    a trial regardless of peer count.
  * Two-rung ASHA promotion (helpers below + orchestrator integration).
    Run every trial to ``max_steps // 4`` first (rung 1) then promote
    the top-50% by rung-1 score to a full-budget rung 2. Default OFF —
    it changes the run set semantically (rung-2 entries override rung-1
    in the ledger for the same config_id).

The killer is exposed as a small ``Pruner`` Protocol so future
strategies (median-stopping, asynchronous successive halving, etc.)
can drop in without touching :mod:`bracket.orchestrator.loop`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Sequence

from bracket.frame import LossFrame

logger = logging.getLogger("bracket.orchestrator.pruner")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartialResult:
    """Snapshot of one trial mid- or post-flight, used by the promotion gate.

    ``smoothed_loss_tail`` is the mean of the last EMA-smoothed losses (or
    the single final value when only one is available). The promotion gate
    ranks trials lower-is-better on this field, mirroring the orchestrator
    scorer's loss component.
    """

    run_id: str
    n_steps_completed: int
    smoothed_loss_tail: float
    grad_norm_p99: Optional[float]
    walltime_s: float

    @classmethod
    def from_frames(
        cls,
        run_id: str,
        frames: Sequence[LossFrame],
        walltime_s: float = 0.0,
        tail_fraction: float = 0.25,
    ) -> "PartialResult":
        n = len(frames)
        if n == 0:
            return cls(
                run_id=run_id,
                n_steps_completed=0,
                smoothed_loss_tail=math.inf,
                grad_norm_p99=None,
                walltime_s=walltime_s,
            )
        window = max(1, int(round(tail_fraction * n)))
        tail = frames[-window:]
        smoothed = [f.smoothed_loss for f in tail if not math.isnan(f.smoothed_loss)]
        tail_mean = sum(smoothed) / len(smoothed) if smoothed else math.inf
        grads = [f.grad_norm for f in frames if f.grad_norm is not None]
        grad_p99: Optional[float] = None
        if grads:
            grads_sorted = sorted(grads)
            idx = min(len(grads_sorted) - 1, int(0.99 * (len(grads_sorted) - 1)))
            grad_p99 = grads_sorted[idx]
        return cls(
            run_id=run_id,
            n_steps_completed=frames[-1].step,
            smoothed_loss_tail=tail_mean,
            grad_norm_p99=grad_p99,
            walltime_s=walltime_s,
        )


# ---------------------------------------------------------------------------
# Pruner protocol
# ---------------------------------------------------------------------------


class Pruner(Protocol):
    """A strategy for early-stopping bad trials and promoting good ones."""

    def should_kill_divergent(
        self, run_id: str, partial_loss_series: Sequence[LossFrame],
    ) -> bool: ...

    def should_promote_to_rung2(
        self, rung1_results: Sequence[PartialResult],
    ) -> set[str]: ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


@dataclass
class DivergenceKiller:
    """Kill obviously-broken trials. Grace period until ``min_peers`` healthy
    trials have completed; afterwards a sigma-test on the smoothed loss vs.
    the per-step-bucket median of peers fires.

    Configuration:
      * ``min_peers`` — number of completed peer trials required before any
        peer-comparison check runs. Defaults to 2.
      * ``sigma_threshold`` — z-score above peer median required to kill.
      * ``min_steps_before_check`` — skip the test for the first N steps,
        the early-training loss is too noisy to compare.
      * ``nan_kill_window`` — kill instantly if any loss frame in the last N
        steps is NaN or +/-inf.

    The killer is **stateless across calls** w.r.t. the live trial: it
    reads ``partial_loss_series`` each tick from the orchestrator.
    Completed-peer state is held internally and updated via
    :meth:`record_completed`.
    """

    sigma_threshold: float = 3.0
    min_peers: int = 2
    min_steps_before_check: int = 25
    nan_kill_window: int = 50
    step_bucket_size: int = 25
    # Internal: bucketed peer values. peer_buckets[bucket_idx] = list of
    # smoothed losses at that bucket from completed peers.
    _peer_buckets: dict[int, list[float]] = field(default_factory=dict)
    _n_peers: int = 0

    def record_completed(self, frames: Sequence[LossFrame]) -> None:
        """Bucket a finished trial's loss curve so the killer can compare
        future trials against it. Trials with NaN finals are skipped — they
        already failed, so they're not useful peer data."""
        if not frames:
            return
        if math.isnan(frames[-1].smoothed_loss):
            return
        self._n_peers += 1
        for f in frames:
            if math.isnan(f.smoothed_loss):
                continue
            bucket = f.step // max(1, self.step_bucket_size)
            self._peer_buckets.setdefault(bucket, []).append(float(f.smoothed_loss))

    @property
    def n_completed_peers(self) -> int:
        return self._n_peers

    def should_kill_divergent(
        self, run_id: str, partial_loss_series: Sequence[LossFrame],
    ) -> bool:
        if not partial_loss_series:
            return False
        # 1. Hard kill: NaN/inf in the most recent window.
        recent = partial_loss_series[-self.nan_kill_window:]
        for f in recent:
            if math.isnan(f.raw_loss) or math.isinf(f.raw_loss):
                logger.warning(
                    "pruner: run %s killed — NaN/inf in raw_loss at step %d",
                    run_id, f.step,
                )
                return True
            if math.isnan(f.smoothed_loss) or math.isinf(f.smoothed_loss):
                logger.warning(
                    "pruner: run %s killed — NaN/inf in smoothed_loss at step %d",
                    run_id, f.step,
                )
                return True
        # 2. Peer-comparison: needs grace period.
        if self._n_peers < self.min_peers:
            return False
        latest = partial_loss_series[-1]
        if latest.step < self.min_steps_before_check:
            return False
        bucket = latest.step // max(1, self.step_bucket_size)
        peer_values = self._peer_buckets.get(bucket)
        if not peer_values:
            # No data at this bucket yet — give the trial benefit of the
            # doubt rather than killing on missing comparison.
            return False
        median, sigma = _median_and_sigma(peer_values)
        if sigma <= 0.0:
            return False
        z = (latest.smoothed_loss - median) / sigma
        if z > self.sigma_threshold:
            logger.warning(
                "pruner: run %s killed — z=%.2f at step %d (loss=%.4g vs peer median %.4g)",
                run_id, z, latest.step, latest.smoothed_loss, median,
            )
            return True
        return False

    def should_promote_to_rung2(
        self, rung1_results: Sequence[PartialResult],
    ) -> set[str]:
        return promote_top_half(rung1_results)


# ---------------------------------------------------------------------------
# Promotion helpers
# ---------------------------------------------------------------------------


def promote_top_half(rung1_results: Sequence[PartialResult]) -> set[str]:
    """Return the set of run_ids whose ``smoothed_loss_tail`` is in the top
    50% (ceiling). Ties broken by ``run_id`` lexicographically so the result
    is deterministic. Trials with non-finite tail loss are never promoted —
    they're already disqualified at rung 1."""
    finite = [r for r in rung1_results if math.isfinite(r.smoothed_loss_tail)]
    if not finite:
        return set()
    finite_sorted = sorted(finite, key=lambda r: (r.smoothed_loss_tail, r.run_id))
    cutoff = max(1, (len(finite_sorted) + 1) // 2)
    return {r.run_id for r in finite_sorted[:cutoff]}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _median_and_sigma(values: Iterable[float]) -> tuple[float, float]:
    vs = sorted(float(v) for v in values if not math.isnan(v))
    n = len(vs)
    if n == 0:
        return (0.0, 0.0)
    if n % 2 == 1:
        median = vs[n // 2]
    else:
        median = 0.5 * (vs[n // 2 - 1] + vs[n // 2])
    if n < 2:
        return (median, 0.0)
    mean = sum(vs) / n
    var = sum((v - mean) ** 2 for v in vs) / (n - 1)
    return (median, math.sqrt(var))
