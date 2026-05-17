"""Round-robin tournament + Bradley-Terry MLE → Elo leaderboard.

Used by the finals stage when ``enable_pairwise_finals=True``. Given a
list of finalist run_ids, prompts, and a configured
:class:`PairwiseJudge`, this module:

  1. Builds the round-robin schedule: every distinct ``(a, b, prompt)``
     ordering with ``a < b`` (paired comparisons are symmetric apart
     from prompt). For ``top_k=5`` × 4 prompts = 40 calls.
  2. Records each verdict as a win/loss/tie.
  3. Fits a Bradley-Terry MLE via the iterative MM (minorisation-
     maximisation) algorithm; ties contribute half a win to each side.
     We hand-roll this rather than depend on ``choix`` so the path is
     test-friendly and dependency-free.
  4. Converts BT strengths to Elo ratings; bootstraps 95% CIs by
     resampling pairs.
"""

from __future__ import annotations

import logging
import math
import random as _random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from bracket.judge.base import PairwiseJudge

logger = logging.getLogger("bracket.proof.pairwise_ranking")


@dataclass(frozen=True)
class PairwiseMatch:
    a: str            # competitor id (run_id or config_id)
    b: str
    prompt: str
    outcome: int      # -1 = a wins, 0 = tie, +1 = b wins
    reason: str = ""


@dataclass(frozen=True)
class EloEntry:
    competitor_id: str
    elo: float
    elo_low: float
    elo_high: float
    n_wins: float        # half-counts ties
    n_losses: float
    n_ties: int


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def build_schedule(
    competitors: Sequence[str], prompts: Sequence[str],
) -> list[tuple[str, str, str]]:
    """Return all ordered triples ``(a, b, prompt)`` with ``a < b`` and one
    entry per prompt. Empty when fewer than 2 competitors."""
    out: list[tuple[str, str, str]] = []
    ids = sorted(set(competitors))
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            for p in prompts:
                out.append((a, b, p))
    return out


def run_tournament(
    competitors: Sequence[str],
    prompts: Sequence[str],
    images_for: dict[str, list[Path]],
    judge: PairwiseJudge,
) -> list[PairwiseMatch]:
    """Execute the round-robin against ``judge`` and return the recorded
    matches. ``images_for`` maps competitor_id → ordered list of sample
    images (one per prompt, same order as ``prompts``).

    When ``images_for`` lacks a competitor or a prompt index, that pair is
    skipped with a warning rather than raising.
    """
    matches: list[PairwiseMatch] = []
    schedule = build_schedule(competitors, prompts)
    for a, b, prompt in schedule:
        try:
            prompt_idx = prompts.index(prompt)
        except ValueError:
            continue
        a_images = images_for.get(a) or []
        b_images = images_for.get(b) or []
        if prompt_idx >= len(a_images) or prompt_idx >= len(b_images):
            logger.warning(
                "tournament: missing image for prompt %r (a=%s, b=%s); skipping",
                prompt, a, b,
            )
            continue
        outcome = judge.judge_pair(a_images[prompt_idx], b_images[prompt_idx], prompt)
        matches.append(PairwiseMatch(a=a, b=b, prompt=prompt, outcome=int(outcome)))
    return matches


# ---------------------------------------------------------------------------
# Bradley-Terry MLE
# ---------------------------------------------------------------------------


def fit_bradley_terry(
    matches: Sequence[PairwiseMatch],
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Fit Bradley-Terry strengths via the iterative MM algorithm.

    Returns a dict ``{competitor_id: strength}`` with ``sum(strengths) = 1``
    so the values are interpretable as a softmax over "skill". Ties are
    awarded as half a win to each side, the standard approximation for
    BT without an explicit tie-aware extension.

    Edge cases:
      - Empty matches → returns ``{}``.
      - A competitor that never plays gets a uniform-prior strength.
      - A competitor with all losses gets a small positive strength
        (clamped at ``1e-4``) to keep the iteration stable.
    """
    if not matches:
        return {}

    ids: set[str] = set()
    for m in matches:
        ids.add(m.a)
        ids.add(m.b)
    n = len(ids)
    if n == 0:
        return {}

    # Win counts: w[i] = sum of "wins" for competitor i, counting ties as 0.5.
    w: dict[str, float] = {x: 0.0 for x in ids}
    # Per-pair count of comparisons: nij[(i, j)] for i != j.
    nij: dict[tuple[str, str], float] = {}
    for m in matches:
        a, b = m.a, m.b
        if a == b:
            continue
        if m.outcome < 0:
            w[a] += 1.0
        elif m.outcome > 0:
            w[b] += 1.0
        else:
            w[a] += 0.5
            w[b] += 0.5
        nij[(a, b)] = nij.get((a, b), 0.0) + 1.0
        nij[(b, a)] = nij.get((b, a), 0.0) + 1.0

    # MM iteration: p_i ← w_i / sum_j (n_ij / (p_i + p_j))
    p: dict[str, float] = {x: 1.0 / n for x in ids}
    for _it in range(int(max_iter)):
        new_p: dict[str, float] = {}
        for i in ids:
            denom = 0.0
            for j in ids:
                if i == j:
                    continue
                count = nij.get((i, j), 0.0)
                if count == 0.0:
                    continue
                denom += count / max(1e-12, p[i] + p[j])
            if denom <= 0.0:
                new_p[i] = max(1e-4, p[i])
            else:
                new_p[i] = max(1e-4, w[i] / denom) if w[i] > 0 else 1e-4
        # Normalise so strengths sum to 1.
        s = sum(new_p.values()) or 1.0
        for i in ids:
            new_p[i] /= s
        # Convergence test.
        delta = max(abs(new_p[i] - p[i]) for i in ids)
        p = new_p
        if delta < tol:
            break
    return p


def strengths_to_elo(
    strengths: dict[str, float],
    *,
    base_elo: float = 1500.0,
) -> dict[str, float]:
    """Map BT strengths to Elo-like ratings centred at ``base_elo``.

    Uses ``Elo = base_elo + 400 * log10(strength / mean_strength)``. Empty
    input returns ``{}``."""
    if not strengths:
        return {}
    mean = sum(strengths.values()) / len(strengths)
    if mean <= 0:
        return {k: base_elo for k in strengths}
    out: dict[str, float] = {}
    for k, s in strengths.items():
        ratio = max(s, 1e-12) / mean
        out[k] = base_elo + 400.0 * math.log10(ratio)
    return out


def bootstrap_elo_cis(
    matches: Sequence[PairwiseMatch],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
    base_elo: float = 1500.0,
) -> list[EloEntry]:
    """Resample matches with replacement and recompute Elo each draw to
    produce per-competitor CIs.

    Output is sorted by point-estimate Elo descending — i.e. the
    leaderboard order. When ``matches`` is empty, returns ``[]``.
    """
    if not matches:
        return []
    rng = _random.Random(seed)
    point_strengths = fit_bradley_terry(matches)
    point_elos = strengths_to_elo(point_strengths, base_elo=base_elo)

    # Per-competitor accumulator.
    samples: dict[str, list[float]] = {k: [] for k in point_elos}
    for _ in range(int(n_resamples)):
        draw = [matches[rng.randrange(len(matches))] for _ in range(len(matches))]
        s = fit_bradley_terry(draw)
        elos = strengths_to_elo(s, base_elo=base_elo)
        for k in samples:
            samples[k].append(elos.get(k, base_elo))

    # Win/loss/tie counts for display.
    n_wins: dict[str, float] = {k: 0.0 for k in point_elos}
    n_losses: dict[str, float] = {k: 0.0 for k in point_elos}
    n_ties: dict[str, int] = {k: 0 for k in point_elos}
    for m in matches:
        if m.outcome < 0:
            n_wins[m.a] = n_wins.get(m.a, 0.0) + 1.0
            n_losses[m.b] = n_losses.get(m.b, 0.0) + 1.0
        elif m.outcome > 0:
            n_wins[m.b] = n_wins.get(m.b, 0.0) + 1.0
            n_losses[m.a] = n_losses.get(m.a, 0.0) + 1.0
        else:
            n_wins[m.a] = n_wins.get(m.a, 0.0) + 0.5
            n_wins[m.b] = n_wins.get(m.b, 0.0) + 0.5
            n_losses[m.a] = n_losses.get(m.a, 0.0) + 0.5
            n_losses[m.b] = n_losses.get(m.b, 0.0) + 0.5
            n_ties[m.a] = n_ties.get(m.a, 0) + 1
            n_ties[m.b] = n_ties.get(m.b, 0) + 1

    alpha = (1.0 - confidence) / 2.0
    entries: list[EloEntry] = []
    for k, point_elo in point_elos.items():
        s = sorted(samples[k])
        low = s[int(alpha * len(s))] if s else point_elo
        high = s[min(len(s) - 1, int((1.0 - alpha) * len(s)))] if s else point_elo
        entries.append(EloEntry(
            competitor_id=k,
            elo=point_elo,
            elo_low=low,
            elo_high=high,
            n_wins=n_wins.get(k, 0.0),
            n_losses=n_losses.get(k, 0.0),
            n_ties=n_ties.get(k, 0),
        ))
    entries.sort(key=lambda e: e.elo, reverse=True)
    return entries


def render_leaderboard_md(entries: Iterable[EloEntry]) -> str:
    """Render a markdown leaderboard table. Empty input returns ''."""
    entries = list(entries)
    if not entries:
        return ""
    lines: list[str] = []
    lines.append("## Pairwise leaderboard (Bradley-Terry → Elo)")
    lines.append("")
    lines.append(
        "Each pair of finalist configs was compared on every shared prompt; "
        "the VLM picked a winner (or tie). Elo ratings derived from a "
        "Bradley-Terry MLE; 95% CIs from 1000 bootstrap resamples of the "
        "matches."
    )
    lines.append("")
    lines.append("| rank | config | elo | 95% CI | wins | losses | ties |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, e in enumerate(entries, 1):
        lines.append(
            f"| {i} | `{e.competitor_id}` | `{e.elo:.1f}` | "
            f"`[{e.elo_low:.1f}, {e.elo_high:.1f}]` | "
            f"`{e.n_wins:.1f}` | `{e.n_losses:.1f}` | `{e.n_ties}` |"
        )
    return "\n".join(lines)
