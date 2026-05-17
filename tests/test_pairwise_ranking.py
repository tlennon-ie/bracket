"""Bradley-Terry + Elo + tournament scheduler tests.

We test the *algorithm* end-to-end with synthetic match outcomes: A
beats everyone, E loses to everyone, and the rank order recovered from
the leaderboard matches the truth.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from bracket.judge.base import PairwiseJudge, PairwiseOutcome
from bracket.proof.pairwise_ranking import (
    EloEntry,
    PairwiseMatch,
    bootstrap_elo_cis,
    build_schedule,
    fit_bradley_terry,
    render_leaderboard_md,
    run_tournament,
    strengths_to_elo,
)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_schedule_builds_distinct_pairs_per_prompt() -> None:
    pairs = build_schedule(["x", "y", "z"], ["p1", "p2"])
    # C(3,2) * 2 prompts = 6 pairs.
    assert len(pairs) == 6
    # Every pair has a < b.
    for a, b, _p in pairs:
        assert a < b


def test_schedule_empty_with_one_competitor() -> None:
    assert build_schedule(["only"], ["p"]) == []


# ---------------------------------------------------------------------------
# Bradley-Terry MLE
# ---------------------------------------------------------------------------


def _generate_matches(
    competitors: list[str], beats: dict[str, set[str]],
) -> list[PairwiseMatch]:
    """For every ordered pair (a, b) with a < b, emit one match.

    Reads ``beats[winner] = {losers}``: if a beats b → outcome=-1 (a wins);
    if b beats a → outcome=1 (b wins); else tie (0).
    """
    out: list[PairwiseMatch] = []
    for i, a in enumerate(competitors):
        for b in competitors[i + 1:]:
            if b in beats.get(a, set()):
                outcome: PairwiseOutcome = -1  # a wins
            elif a in beats.get(b, set()):
                outcome = 1  # b wins
            else:
                outcome = 0
            out.append(PairwiseMatch(a=a, b=b, prompt="p", outcome=outcome))
    return out


def test_bt_recovers_total_order() -> None:
    """A>B>C>D>E (each higher beats each lower)."""
    comps = ["A", "B", "C", "D", "E"]
    beats: dict[str, set[str]] = {
        "A": {"B", "C", "D", "E"},
        "B": {"C", "D", "E"},
        "C": {"D", "E"},
        "D": {"E"},
    }
    # Multiple matches per pair so MLE has signal.
    matches = []
    for _ in range(5):
        matches.extend(_generate_matches(comps, beats))
    strengths = fit_bradley_terry(matches)
    # Strict descending order.
    ranked = sorted(strengths.keys(), key=lambda k: strengths[k], reverse=True)
    assert ranked == comps


def test_bt_handles_empty_matches() -> None:
    assert fit_bradley_terry([]) == {}


def test_strengths_to_elo_centred_at_base() -> None:
    """Equal strengths → all Elos equal to base."""
    strengths = {"a": 0.5, "b": 0.5}
    elos = strengths_to_elo(strengths, base_elo=1500.0)
    assert elos["a"] == pytest.approx(1500.0)
    assert elos["b"] == pytest.approx(1500.0)


def test_strengths_to_elo_handles_zero() -> None:
    elos = strengths_to_elo({"a": 0.0, "b": 1.0}, base_elo=1500.0)
    # Zero strength → very low Elo, but finite.
    assert elos["a"] < elos["b"]
    assert math.isfinite(elos["a"])


# ---------------------------------------------------------------------------
# End-to-end leaderboard
# ---------------------------------------------------------------------------


def test_bootstrap_leaderboard_orders_total_order_correctly() -> None:
    comps = ["A", "B", "C", "D", "E"]
    beats: dict[str, set[str]] = {
        "A": {"B", "C", "D", "E"},
        "B": {"C", "D", "E"},
        "C": {"D", "E"},
        "D": {"E"},
    }
    matches = []
    for _ in range(5):
        matches.extend(_generate_matches(comps, beats))
    entries = bootstrap_elo_cis(matches, n_resamples=100, seed=42)
    order = [e.competitor_id for e in entries]
    assert order == comps
    # Top should have higher Elo than bottom.
    assert entries[0].elo > entries[-1].elo
    # CIs must be ordered low <= elo <= high.
    for e in entries:
        assert e.elo_low <= e.elo <= e.elo_high


def test_render_leaderboard_md_includes_all_entries() -> None:
    entries = [
        EloEntry(competitor_id="A", elo=1600.0, elo_low=1550.0, elo_high=1650.0,
                 n_wins=4, n_losses=0, n_ties=0),
        EloEntry(competitor_id="B", elo=1400.0, elo_low=1350.0, elo_high=1450.0,
                 n_wins=0, n_losses=4, n_ties=0),
    ]
    md = render_leaderboard_md(entries)
    assert "Pairwise leaderboard" in md
    assert "`A`" in md
    assert "`B`" in md
    assert "1600.0" in md
    assert "1400.0" in md


def test_render_leaderboard_md_empty() -> None:
    assert render_leaderboard_md([]) == ""


# ---------------------------------------------------------------------------
# Tournament runner with a stub judge
# ---------------------------------------------------------------------------


class _StubPairwiseJudge:
    """Deterministic stub: always says the alphabetically-earlier id wins."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def judge_pair(
        self, image_a: Path, image_b: Path, prompt: str,
    ) -> PairwiseOutcome:
        self.calls.append((image_a.name, image_b.name, prompt))
        # Names embed the competitor id: "A_p0.png" etc. Earlier letter
        # always wins.
        a_id = image_a.name.split("_")[0]
        b_id = image_b.name.split("_")[0]
        if a_id < b_id:
            return -1
        if a_id > b_id:
            return 1
        return 0

    def eject(self) -> None: ...


def test_run_tournament_records_matches_per_prompt(tmp_path: Path) -> None:
    comps = ["A", "B", "C"]
    prompts = ["p0", "p1"]
    images_for: dict[str, list[Path]] = {}
    for c in comps:
        paths = []
        for p in prompts:
            f = tmp_path / f"{c}_{p}.png"
            f.write_bytes(b"\x89PNG\r\n")
            paths.append(f)
        images_for[c] = paths
    judge: PairwiseJudge = _StubPairwiseJudge()  # type: ignore[assignment]
    matches = run_tournament(comps, prompts, images_for, judge)
    # C(3,2) * 2 = 6 matches.
    assert len(matches) == 6
    # All "earlier letter wins" → outcome = -1 (a wins) for every pair.
    for m in matches:
        assert m.outcome == -1


def test_run_tournament_skips_missing_images_gracefully(tmp_path: Path) -> None:
    comps = ["A", "B"]
    prompts = ["p"]
    images_for: dict[str, list[Path]] = {
        "A": [tmp_path / "A_p.png"],
        # B has no images at all.
        "B": [],
    }
    images_for["A"][0].write_bytes(b"\x89PNG")
    judge: PairwiseJudge = _StubPairwiseJudge()  # type: ignore[assignment]
    matches = run_tournament(comps, prompts, images_for, judge)
    assert matches == []
