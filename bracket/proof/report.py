"""Markdown comparison report with statistical confidence on the verdict.

When `seeds_per_config > 1`, multiple seed runs per config let us compute
Welch's t-test between best and runner-up — and report the result honestly
("high confidence", "marginal", "noisy — extend budget").

Sections:
  - Headline: orchestrator vs baseline, with confidence
  - Top-K configs (aggregated across seeds): mean ± std
  - Sample-quality breakdown when judge ran
  - Disqualifications
  - Score distribution histogram
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from bracket.orchestrator.ledger import Ledger


def generate_report(ledger_path: Path, out_path: Path) -> Path:
    rows = list(Ledger(ledger_path).iter_rows())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = _render(rows)
    out_path.write_text(md, encoding="utf-8")
    return out_path


def _render(rows: list[dict]) -> str:
    if not rows:
        return "# Bracket · orchestration report\n\nNo runs in ledger.\n"

    # Group only training rows by config_id. Setup rows have empty configs
    # and would otherwise inflate the unique-configs count by 1.
    _training_roles = ("baseline", "candidate", "curated", "finalist")
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("role") not in _training_roles:
            continue
        cid = r.get("config_id") or _fallback_config_id(r.get("config", {}))
        by_cfg[cid].append(r)

    aggregates: list[dict] = []
    for cid, group in by_cfg.items():
        scored = [g for g in group if g.get("score") is not None]
        if not scored:
            aggregates.append({
                "config_id": cid, "role": group[0].get("role"),
                "config": group[0].get("config", {}),
                "n_seeds": len(group), "n_scored": 0,
                "mean": None, "stdev": None, "min": None,
                "first_run_id": group[0].get("run_id"),
                "any_disqualified": True,
                "judge": None,
                "score_components": group[0].get("score_components", {}),
            })
            continue
        scores = [g["score"] for g in scored]
        mean = statistics.fmean(scores)
        sd = statistics.stdev(scores) if len(scores) >= 2 else 0.0
        # Surface a representative judge_report (mean across seeds) if present
        judge_means = _mean_judge(scored)
        aggregates.append({
            "config_id": cid, "role": group[0].get("role"),
            "config": group[0].get("config", {}),
            "n_seeds": len(group), "n_scored": len(scored),
            "mean": mean, "stdev": sd, "min": min(scores),
            "first_run_id": group[0].get("run_id"),
            "any_disqualified": any(g.get("disqualified") for g in group),
            "judge": judge_means,
            "score_components": scored[-1].get("score_components", {}),
            "_scores": scores,
        })

    baseline_agg = next((a for a in aggregates if a["role"] == "baseline"), None)
    candidate_aggs = [a for a in aggregates if a["role"] != "baseline" and a["mean"] is not None]
    scored_aggs = [a for a in aggregates if a["mean"] is not None]
    disqualified_runs = [r for r in rows if r.get("disqualified")]
    sorted_aggs = sorted(scored_aggs, key=lambda a: a["mean"]) if scored_aggs else []
    best = sorted_aggs[0] if sorted_aggs else None
    runner_up = sorted_aggs[1] if len(sorted_aggs) >= 2 else None

    # Training-only counts (exclude setup rows so the totals match what the
    # user sees on the dashboard).
    training_rows = [r for r in rows if r.get("role") in ("baseline", "candidate", "curated", "finalist")]
    rows_with_judge = [r for r in training_rows if r.get("judge_report")]

    lines: list[str] = []
    lines.append("# Bracket · orchestration report")
    lines.append("")
    lines.append(f"- Training runs: **{len(training_rows)}** "
                 f"(unique configs: {len(by_cfg)}, scored configs: {len(scored_aggs)}, "
                 f"disqualified runs: {len(disqualified_runs)})")
    multi_seed = any(a["n_scored"] >= 2 for a in scored_aggs)
    if multi_seed:
        max_seeds = max(a["n_scored"] for a in scored_aggs)
        lines.append(f"- Multi-seed: up to **{max_seeds}** seeds per config — confidence intervals computed")
    else:
        lines.append("- Single-seed: confidence interval computation skipped (need ≥ 2 seeds per config)")
    if not training_rows:
        pass
    elif not rows_with_judge:
        lines.append("- **Visual scoring:** ❌ judge not configured — runs ranked on training loss only.")
    else:
        lines.append(f"- **Visual scoring:** ✓ {len(rows_with_judge)}/{len(training_rows)} runs judged by LMStudio.")

    # Plain-language reading guide. New users see numbers without context;
    # this paragraph tells them what each column means and how the winner
    # was picked.
    lines.append("")
    lines.append("## How to read this report")
    lines.append("")
    lines.append(
        "- **score (lower = better)**: a single number summarising each run. "
        "Computed as `final_smoothed_loss + |slope_of_last_25%|` — i.e. how low the "
        "loss curve ended **plus** how flat it was at the end. A run that ended at "
        "loss 0.30 with a flat curve scores ~0.30; one that ended at 0.30 but was "
        "still steeply diverging scores higher (worse)."
    )
    lines.append(
        "- **mean**: the average score across all seeds run for the same config. "
        "With 1 seed, mean equals that single run's score. With ≥2 seeds, it averages "
        "out per-seed luck so we're comparing configs, not random initialisations."
    )
    lines.append(
        "- **stdev**: how much the score varied across seeds for the same config. "
        "stdev=0 means single-seed (no variation measurable). Small stdev = "
        "**consistent** config — different random seeds produce similar results. "
        "Large stdev = **noisy** config — keep more seeds before trusting it."
    )
    lines.append(
        "- **n_seeds**: how many independent runs of this config we have. More = "
        "more trustworthy ranking. Single-seed comparisons can flip with one extra "
        "seed; multi-seed rankings are statistically meaningful."
    )
    lines.append(
        "- **The winner** below is whichever config has the lowest **mean** score "
        "and isn't disqualified. With single-seed, that's the run with the lowest "
        "raw score. With multi-seed, we also show p-values and 95% confidence intervals."
    )

    lines.append("")
    lines.append("## Headline")
    lines.append("")

    if baseline_agg is None:
        lines.append("_No baseline in this session._")
    elif best is None:
        lines.append("_No scored configs — every candidate was disqualified._")
    elif baseline_agg["mean"] is None:
        lines.append(f"**Baseline disqualified.** Best candidate: `{best['first_run_id']}` "
                     f"score=`{best['mean']:.6f}`.")
    elif best["config_id"] == baseline_agg["config_id"]:
        lines.append(f"**Baseline still wins.** Score: `{baseline_agg['mean']:.6f}`. "
                     f"Of {len(candidate_aggs)} candidates, none beat it.")
    else:
        delta = best["mean"] - baseline_agg["mean"]
        pct = (delta / baseline_agg["mean"]) * 100 if baseline_agg["mean"] else 0.0
        verdict_line = "**Orchestrator beat baseline.**" if delta < 0 else "**Baseline still wins.**"
        lines.append(verdict_line)
        # Plain-English summary — same numbers in a sentence.
        if delta < 0:
            lines.append("")
            lines.append(
                f"The best config landed at score **{best['mean']:.4f}** vs the baseline's "
                f"**{baseline_agg['mean']:.4f}**, a **{abs(pct):.2f}% improvement**. "
                f"Lower is better — see [How to read this report](#how-to-read-this-report) above."
            )
        else:
            lines.append("")
            lines.append(
                f"None of the {len(candidate_aggs)} candidates beat the baseline's score of "
                f"**{baseline_agg['mean']:.4f}**. The closest got to "
                f"**{best['mean']:.4f}** ({abs(pct):.2f}% worse)."
            )
        lines.append("")
        lines.append("| | run_id | mean | stdev | n_seeds |")
        lines.append("|---|---|---|---|---|")
        lines.append(_agg_row("baseline", baseline_agg))
        lines.append(_agg_row("**best**", best))
        lines.append("")
        lines.append(f"Δ score: `{delta:+.6f}` ({pct:+.2f}%)")

        # Confidence (Welch's t-test) — only meaningful with multi-seed
        if best["n_scored"] >= 2 and baseline_agg["n_scored"] >= 2:
            t_p = welch_t_test(best["_scores"], baseline_agg["_scores"])
            ci_low, ci_high = welch_ci_diff(best["_scores"], baseline_agg["_scores"])
            verdict = _confidence_verdict(t_p, delta, ci_low, ci_high)
            lines.append("")
            lines.append("**Confidence (best vs baseline):**")
            lines.append(f"- Welch's t-test p-value: `{t_p:.4g}`")
            lines.append(f"- 95% CI for Δ score: `[{ci_low:+.6f}, {ci_high:+.6f}]`")
            lines.append(f"- Verdict: {verdict}")

    # Confidence (best vs runner-up) — also only when multi-seed
    if best is not None and runner_up is not None:
        if best["n_scored"] >= 2 and runner_up["n_scored"] >= 2:
            t_p = welch_t_test(best["_scores"], runner_up["_scores"])
            ci_low, ci_high = welch_ci_diff(best["_scores"], runner_up["_scores"])
            delta = best["mean"] - runner_up["mean"]
            verdict = _confidence_verdict(t_p, delta, ci_low, ci_high)
            lines.append("")
            lines.append("**Confidence (best vs runner-up):**")
            lines.append(f"- Δ mean: `{delta:+.6f}`")
            lines.append(f"- Welch's t-test p-value: `{t_p:.4g}`")
            lines.append(f"- 95% CI for Δ score: `[{ci_low:+.6f}, {ci_high:+.6f}]`")
            lines.append(f"- Verdict: {verdict}")

    # Top-K
    lines.append("")
    lines.append("## Top-5 configs (aggregated by config_id)")
    lines.append("")
    if not sorted_aggs:
        lines.append("_None scored._")
    else:
        lines.append("| rank | role | config_id | mean | stdev | n_seeds | knobs |")
        lines.append("|---|---|---|---|---|---|---|")
        for i, a in enumerate(sorted_aggs[:5], 1):
            lines.append(
                f"| {i} | {a['role']} | `{a['config_id']}` | "
                f"`{_fmt(a['mean'])}` | `{_fmt(a['stdev'])}` | {a['n_scored']} | "
                f"{_compact_knobs(a['config'])} |"
            )

    # Sample quality breakdown
    if any(a.get("judge") for a in scored_aggs):
        lines.append("")
        lines.append("## Sample quality (VLM judge)")
        lines.append("")
        lines.append("Higher is better. Scores are 0–10 means across each config's seeds.")
        lines.append("")
        lines.append("| rank | config_id | overall | adherence | quality | artifact-free |")
        lines.append("|---|---|---|---|---|---|")
        with_judge = [a for a in sorted_aggs if a.get("judge") is not None]
        for i, a in enumerate(with_judge[:5], 1):
            j = a["judge"]
            lines.append(
                f"| {i} | `{a['config_id']}` | "
                f"`{j['mean_overall']:.2f}` | `{j['mean_prompt_adherence']:.2f}` | "
                f"`{j['mean_visual_quality']:.2f}` | `{j['mean_artifact_free']:.2f}` |"
            )

    # Disqualifications
    if disqualified_runs:
        lines.append("")
        lines.append("## Disqualifications")
        lines.append("")
        reasons = Counter(r.get("disqualified", "?") for r in disqualified_runs)
        for reason, count in reasons.most_common():
            lines.append(f"- `{reason}`: {count}")

    # Self-consistency uncertainty: runs where the VLM disagreed with
    # itself across n_samples > 1. Surfaced so the user can spot-check.
    uncertain_rows = [r for r in training_rows if r.get("judge_uncertain")]
    if uncertain_rows:
        lines.append("")
        lines.append("## Uncertain judgements")
        lines.append("")
        lines.append(
            "The VLM disagreed with itself across `judge_n_samples` calls "
            "on at least one image in these runs (score variance crossed the "
            "configured threshold). Worth a manual look:"
        )
        lines.append("")
        for r in uncertain_rows[:10]:
            lines.append(
                f"- `{r.get('run_id')}` "
                f"({r.get('judge_report', {}).get('n_uncertain', '?')} "
                f"uncertain / "
                f"{r.get('judge_report', {}).get('n_images', '?')} images)"
            )

    # Score distribution
    if scored_aggs:
        lines.append("")
        lines.append("## Score distribution (mean per config)")
        lines.append("")
        lines.append("```")
        lines.append(_ascii_histogram([a["mean"] for a in scored_aggs]))
        lines.append("```")

    return "\n".join(lines) + "\n"


# ─────────────────────────── helpers ───────────────────────────


def _agg_row(label: str, a: dict) -> str:
    return (f"| {label} | `{a['first_run_id']}` | "
            f"`{_fmt(a['mean'])}` | `{_fmt(a['stdev'])}` | {a['n_scored']} |")


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def _compact_knobs(c: dict) -> str:
    short_keys = ("learning_rate", "optimizer_type", "lr_scheduler", "lr_warmup_steps",
                  "network_dim", "network_alpha", "noise_offset", "train_batch_size")
    parts = []
    for k in short_keys:
        if k in c:
            v = c[k]
            parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return "; ".join(parts)


def _fallback_config_id(config: dict) -> str:
    import hashlib
    import json as _json
    return hashlib.sha1(_json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _mean_judge(scored: list[dict]) -> Optional[dict]:
    judges = [g.get("judge_report") for g in scored if g.get("judge_report")]
    if not judges:
        return None
    keys = ("mean_overall", "mean_prompt_adherence", "mean_visual_quality", "mean_artifact_free")
    return {k: statistics.fmean(j[k] for j in judges if k in j) for k in keys}


def _ascii_histogram(values: list[float], bins: int = 10, width: int = 40) -> str:
    if not values:
        return "(empty)"
    lo, hi = min(values), max(values)
    if lo == hi:
        return f"all values = {lo:.6f}"
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[idx] += 1
    max_count = max(counts) or 1
    rows = []
    for i in range(bins):
        bar = "#" * int(round(width * counts[i] / max_count))
        rows.append(f"{edges[i]:8.4f}–{edges[i+1]:8.4f} | {counts[i]:3d} | {bar}")
    return "\n".join(rows)


def welch_t_test(a: list[float], b: list[float]) -> float:
    """Two-sided Welch's t-test, returning the p-value. Pure-python so no scipy
    dependency. Survives tiny samples (n=2 each side) — the regime we're in."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    if va == 0 and vb == 0:
        return 0.0 if ma == mb else 0.0
    se2 = va / na + vb / nb
    if se2 == 0:
        return 1.0 if ma == mb else 0.0
    t = (ma - mb) / math.sqrt(se2)
    # Welch–Satterthwaite degrees of freedom
    df_num = (va / na + vb / nb) ** 2
    df_den = (va ** 2) / (na ** 2 * (na - 1)) + (vb ** 2) / (nb ** 2 * (nb - 1))
    df = df_num / df_den if df_den > 0 else (na + nb - 2)
    return _two_sided_p_from_t(abs(t), df)


def welch_ci_diff(a: list[float], b: list[float], level: float = 0.95) -> tuple[float, float]:
    """95% CI for (mean(a) - mean(b)) under Welch's approximation."""
    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    df_num = (va / na + vb / nb) ** 2
    df_den = (va ** 2) / (na ** 2 * (na - 1)) + (vb ** 2) / (nb ** 2 * (nb - 1))
    df = df_num / df_den if df_den > 0 else (na + nb - 2)
    # Critical t — 2.776 for df=4 95%; we approximate via inverse incomplete-beta.
    # For our small-N regime it's enough to use a coarse table lookup.
    t_crit = _t_critical_two_sided(level, df)
    margin = t_crit * se
    diff = ma - mb
    return (diff - margin, diff + margin)


def _two_sided_p_from_t(t: float, df: float) -> float:
    """Two-sided p-value from t-statistic. Uses regularised incomplete beta."""
    if df <= 0 or math.isinf(t) or math.isnan(t):
        return float("nan")
    x = df / (df + t * t)
    return _incomplete_beta(x, df / 2.0, 0.5)


def _t_critical_two_sided(level: float, df: float) -> float:
    # Coarse but adequate table lookup for level=0.95 (the only level we use).
    if abs(level - 0.95) > 1e-6:
        # generic approximation good for df >= 1
        return 1.96 + (2.776 - 1.96) * max(0.0, min(1.0, (5 - df) / 4))
    table = [
        (1, 12.706), (2, 4.303), (3, 3.182), (4, 2.776), (5, 2.571),
        (6, 2.447), (8, 2.306), (10, 2.228), (12, 2.179), (16, 2.120),
        (20, 2.086), (30, 2.042), (60, 2.000), (120, 1.980), (1000, 1.962),
    ]
    for cutoff_df, val in table:
        if df <= cutoff_df:
            return val
    return 1.96


# Continued-fraction implementation of the regularised incomplete beta
# function I_x(a, b). Standard NR recipe; good to ~1e-7 in our N regime.
def _incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(x, a, b) / a
    return 1.0 - bt * _betacf(1.0 - x, b, a) / b


def _betacf(x: float, a: float, b: float, max_iter: int = 200, eps: float = 3.0e-7) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def _confidence_verdict(p: float, delta: float, ci_low: float, ci_high: float) -> str:
    """Translate p-value + CI into a one-line plain-English verdict."""
    if math.isnan(p):
        return "**insufficient data** — need ≥2 seeds per config"
    if delta >= 0:
        return "best is *not* better than this comparator on the mean"
    if p < 0.01:
        return "**high confidence** (p<0.01)"
    if p < 0.05:
        return "moderate confidence (p<0.05)"
    if p < 0.20:
        return "marginal — would benefit from more seeds"
    return "**noisy** — extend budget / seeds to confirm"
