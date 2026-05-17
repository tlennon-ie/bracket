"""Unit tests for the divergence killer + 2-rung ASHA promotion gate."""
from __future__ import annotations

import math

import pytest

from bracket.frame import LossFrame
from bracket.orchestrator.pruner import (
    DivergenceKiller,
    PartialResult,
    promote_top_half,
)


def _frame(step: int, raw: float, smoothed: float | None = None) -> LossFrame:
    return LossFrame(
        step=step,
        raw_loss=raw,
        smoothed_loss=raw if smoothed is None else smoothed,
        lr=None,
        wall_time=float(step),
        timestep=None,
        timestep_bucket=None,
        grad_norm=None,
    )


# ---------------------------------------------------------------------------
# DivergenceKiller — NaN/inf hard kill
# ---------------------------------------------------------------------------


def test_killer_hard_kills_on_nan_loss() -> None:
    killer = DivergenceKiller()
    series = [_frame(s, 0.5) for s in range(1, 30)]
    series.append(_frame(30, float("nan"), float("nan")))
    assert killer.should_kill_divergent("r1", series) is True


def test_killer_hard_kills_on_inf_loss() -> None:
    killer = DivergenceKiller()
    series = [_frame(s, 0.5) for s in range(1, 30)]
    series.append(_frame(30, float("inf"), 0.5))
    assert killer.should_kill_divergent("r1", series) is True


def test_killer_does_not_kill_clean_run_without_peers() -> None:
    """First trial has no peers → no peer-comparison kill, no NaN → no kill."""
    killer = DivergenceKiller()
    series = [_frame(s, 0.5 - s * 0.001) for s in range(1, 100)]
    assert killer.should_kill_divergent("r1", series) is False


# ---------------------------------------------------------------------------
# DivergenceKiller — peer-comparison after grace period
# ---------------------------------------------------------------------------


def test_killer_activates_after_two_completed_peers() -> None:
    """A diverging trial passes inspection with 0 peers, fires once peers
    have completed and there's a clear-enough divergence."""
    killer = DivergenceKiller(sigma_threshold=3.0, min_steps_before_check=10)
    # Two healthy peers around loss=0.3.
    for run_id in ("peer1", "peer2"):
        peer = [_frame(s, 0.3 + 0.01 * ((-1) ** s)) for s in range(1, 200)]
        # Quick sanity: not killed.
        assert killer.should_kill_divergent(run_id, peer) is False
        killer.record_completed(peer)
    assert killer.n_completed_peers == 2

    # Now a clearly-diverging third trial: loss ~5x peer median.
    bad = [_frame(s, 1.5) for s in range(1, 80)]
    assert killer.should_kill_divergent("bad", bad) is True


def test_killer_respects_grace_window_before_first_check() -> None:
    """Even with peer data, killer skips trials still in their grace window."""
    killer = DivergenceKiller(
        sigma_threshold=3.0, min_steps_before_check=100,
    )
    for run_id in ("p1", "p2"):
        killer.record_completed([_frame(s, 0.3) for s in range(1, 200)])
    early = [_frame(s, 5.0) for s in range(1, 50)]  # diverging but step<100
    assert killer.should_kill_divergent("c", early) is False


def test_killer_does_not_kill_trial_inside_peer_distribution() -> None:
    """A trial that tracks the peer distribution must not be killed."""
    killer = DivergenceKiller(sigma_threshold=3.0)
    for run_id in ("p1", "p2", "p3"):
        killer.record_completed([_frame(s, 0.3 + 0.005 * ((-1) ** s)) for s in range(1, 200)])
    healthy = [_frame(s, 0.30 + 0.01) for s in range(1, 80)]
    assert killer.should_kill_divergent("c", healthy) is False


def test_killer_ignores_completed_peer_with_nan_final() -> None:
    killer = DivergenceKiller()
    nan_run = [_frame(s, 0.3) for s in range(1, 50)]
    nan_run.append(_frame(50, float("nan"), float("nan")))
    killer.record_completed(nan_run)
    assert killer.n_completed_peers == 0


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------


def _partial(run_id: str, tail: float, steps: int = 100) -> PartialResult:
    return PartialResult(
        run_id=run_id,
        n_steps_completed=steps,
        smoothed_loss_tail=tail,
        grad_norm_p99=None,
        walltime_s=10.0,
    )


def test_promote_top_half_returns_lower_half_by_score() -> None:
    results = [
        _partial("r0", 0.10),
        _partial("r1", 0.20),
        _partial("r2", 0.30),
        _partial("r3", 0.40),
        _partial("r4", 0.50),
        _partial("r5", 0.60),
    ]
    promoted = promote_top_half(results)
    assert promoted == {"r0", "r1", "r2"}


def test_promote_top_half_with_odd_count_promotes_ceiling() -> None:
    results = [
        _partial("r0", 0.10),
        _partial("r1", 0.20),
        _partial("r2", 0.30),
    ]
    promoted = promote_top_half(results)
    # ceil(3/2) = 2.
    assert promoted == {"r0", "r1"}


def test_promote_top_half_skips_disqualified() -> None:
    results = [
        _partial("r0", math.inf),
        _partial("r1", 0.20),
        _partial("r2", 0.30),
        _partial("r3", 0.40),
    ]
    promoted = promote_top_half(results)
    # 3 finite trials, top half → top 2 (ceil(3/2))
    assert promoted == {"r1", "r2"}


def test_promote_top_half_empty() -> None:
    assert promote_top_half([]) == set()


def test_promote_top_half_six_trials_returns_three() -> None:
    """The user-doc'd contract: 6 trials → top 3 promoted."""
    results = [_partial(f"r{i}", 0.1 + 0.05 * i) for i in range(6)]
    promoted = promote_top_half(results)
    assert promoted == {"r0", "r1", "r2"}


# ---------------------------------------------------------------------------
# PartialResult.from_frames
# ---------------------------------------------------------------------------


def test_partial_result_from_frames_uses_tail_window() -> None:
    """tail_fraction=0.25 of 100 frames = 25 → mean of last 25."""
    frames = [_frame(s, 1.0, 1.0) for s in range(1, 76)] + \
             [_frame(s, 0.10, 0.10) for s in range(76, 101)]
    pr = PartialResult.from_frames("r", frames, walltime_s=12.5, tail_fraction=0.25)
    assert pr.run_id == "r"
    assert pr.n_steps_completed == 100
    assert pr.walltime_s == 12.5
    # All 25 tail frames are 0.10.
    assert pr.smoothed_loss_tail == pytest.approx(0.10, abs=1e-6)


def test_partial_result_from_frames_handles_empty() -> None:
    pr = PartialResult.from_frames("r", [])
    assert pr.n_steps_completed == 0
    assert math.isinf(pr.smoothed_loss_tail)


# ---------------------------------------------------------------------------
# ASHA wrapper helpers
# ---------------------------------------------------------------------------


def test_asha_rung1_steps_floored_for_tiny_budget() -> None:
    from bracket.orchestrator.asha import rung1_max_steps
    assert rung1_max_steps(400) == 100
    assert rung1_max_steps(100) == 25
    # Floor at 25 even when 1/4 budget would be smaller.
    assert rung1_max_steps(40) == 25
    assert rung1_max_steps(4) == 25


def test_asha_resume_supported_list_covers_known_trainers() -> None:
    from bracket.orchestrator.asha import trainer_supports_resume
    assert trainer_supports_resume("sdxl-sd-scripts") is True
    assert trainer_supports_resume("zimage-lora-musubi") is True
    assert trainer_supports_resume("flux2-klein-lora-musubi") is True
    assert trainer_supports_resume("nonexistent-trainer") is False
    # Catch typos in the curated list: there are 19 trainer adapters.
    from bracket.orchestrator.asha import _RESUME_SUPPORTED_TRAINERS
    assert len(_RESUME_SUPPORTED_TRAINERS) == 19


def test_select_promoted_from_rung1_picks_top_half() -> None:
    """Given six rung-1 entries with varying scores, three best promote."""
    from bracket.orchestrator.asha import select_promoted_from_rung1
    from bracket.orchestrator.loop import config_id
    from bracket.search.controller import LedgerEntry

    history = []
    configs = []
    for i in range(6):
        cfg = {"lr": 1e-4 + i * 1e-5}
        configs.append(cfg)
        history.append(LedgerEntry(
            run_id=f"cand-{i:03d}-s0",
            config=cfg,
            score=0.1 + 0.05 * i,
            error=None,
            duration_s=10.0,
            n_steps=80,
            timestamp=float(i),
        ))
    promoted = select_promoted_from_rung1(history)
    # Best 3 = configs 0, 1, 2.
    expected = {config_id(c) for c in configs[:3]}
    assert promoted == expected


def test_select_promoted_skips_disqualified() -> None:
    from bracket.orchestrator.asha import select_promoted_from_rung1
    from bracket.orchestrator.loop import config_id
    from bracket.search.controller import LedgerEntry

    configs = [{"lr": 1e-4 + i * 1e-5} for i in range(4)]
    history = [
        LedgerEntry(
            run_id="r0", config=configs[0], score=None,  # disqualified
            error="diverging", duration_s=1.0, n_steps=10, timestamp=0.0,
        ),
        LedgerEntry(
            run_id="r1", config=configs[1], score=0.20,
            error=None, duration_s=10.0, n_steps=80, timestamp=1.0,
        ),
        LedgerEntry(
            run_id="r2", config=configs[2], score=0.30,
            error=None, duration_s=10.0, n_steps=80, timestamp=2.0,
        ),
        LedgerEntry(
            run_id="r3", config=configs[3], score=0.40,
            error=None, duration_s=10.0, n_steps=80, timestamp=3.0,
        ),
    ]
    promoted = select_promoted_from_rung1(history)
    # 3 scored → ceil(3/2)=2 promoted. configs 1 and 2 are best.
    expected = {config_id(configs[1]), config_id(configs[2])}
    assert promoted == expected
