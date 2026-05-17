"""Helpers for the Monitor tab — extracts meaningful data from a session
output_dir for the UI to render.

Three things the user actually wants to see, instead of raw log dumps:

  1. Progress card: current candidate N/total, current step X/Y, ETA
  2. Live loss: raw + EMA-smoothed for the *currently running* candidate
  3. Score history: one row per completed candidate, sortable by score

This module reads from disk (ledger, tfevents, sample dirs) — no internal
state. Safe to call frequently from a UI poll.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from bracket.ema import EMASmoother
from bracket.tfevents_reader import TFEventsTail


@dataclass
class CandidateRow:
    """One row in the score-history table (display-friendly)."""
    run_id: str
    role: str
    config_id: str
    score: Optional[float]
    final_smoothed: Optional[float]
    slope: Optional[float]
    n_steps: int
    duration_s: float
    disqualified: Optional[str]


@dataclass
class LossSeries:
    steps: list[int]
    raw: list[float]
    smoothed: list[float]
    # Per-step wall-clock from tfevents (seconds since epoch). Same length
    # as `steps` when populated. Optional for backwards compatibility with
    # older callers that build the series by hand.
    wall_times: list[float] = field(default_factory=list)


@dataclass
class MonitorSnapshot:
    status_line: str
    progress_pct: float          # 0..100, based on completed runs / target
    completed_runs: int
    total_runs_target: int
    current_run_id: Optional[str]
    current_run_steps_done: Optional[int]
    current_run_max_steps: Optional[int]
    current_loss: Optional[LossSeries]
    score_history: list[CandidateRow]
    setup_status: str            # "not started" | "running" | "done"
    judge_summary: str = ""      # plain-English judge status, e.g. "not configured"
    session_done: bool = False   # True when no run is in-flight and target reached
    # Average iterations-per-second over the last ~10 step deltas of the
    # currently-running run. None when there's nothing in-flight or we
    # don't yet have enough samples to compute a rate.
    current_steps_per_sec: Optional[float] = None
    # Pruner indicators. ``killed_by_pruner`` counts trials the divergence
    # killer stopped early; ``running`` counts in-flight trials (0 or 1
    # for the current single-run-at-a-time orchestrator).
    killed_by_pruner: int = 0
    running_runs: int = 0


def read_ledger(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    rows: list[dict] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_running_run(output_dir: Path, ledger_rows: list[dict]) -> Optional[Path]:
    """Return the run directory whose loss curve we should plot.

    Preference: an in-flight run (dir exists but not yet in ledger). Falls
    back to the most-recently-modified completed run so the loss plot still
    shows useful data when no candidate is currently training (e.g. between
    finals stages, or after the whole session finished).
    """
    runs_dir = Path(output_dir) / "runs"
    if not runs_dir.exists():
        return None
    all_run_dirs = [
        p for p in runs_dir.iterdir()
        if p.is_dir() and p.name != "setup"
    ]
    if not all_run_dirs:
        return None
    ledgered = {r.get("run_id") for r in ledger_rows}
    in_flight = [p for p in all_run_dirs if p.name not in ledgered]
    if in_flight:
        return max(in_flight, key=lambda p: p.stat().st_mtime)
    # Fall back to the latest completed run so the loss chart isn't blank
    # the moment training finishes.
    return max(all_run_dirs, key=lambda p: p.stat().st_mtime)


def find_tfevents_in(run_dir: Path) -> Optional[Path]:
    matches = list(run_dir.rglob("events.out.tfevents.*"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_size if p.exists() else 0)


def load_loss_series(tfevents: Path, *, ema_alpha: float = 0.05) -> Optional[LossSeries]:
    """Drain a tfevents file and return the raw/smoothed loss series."""
    if not tfevents.exists():
        return None
    smoother = EMASmoother(alpha=ema_alpha)
    steps: list[int] = []
    raw: list[float] = []
    smoothed: list[float] = []
    wall_times: list[float] = []

    def _on_frame(frame) -> None:
        steps.append(frame.step)
        raw.append(frame.raw_loss)
        smoothed.append(frame.smoothed_loss)
        wall_times.append(frame.wall_time)

    tail = TFEventsTail(event_path=str(tfevents), on_frame=_on_frame, ema_alpha=ema_alpha)
    n = tail.drain_once()
    if n == 0:
        return None
    return LossSeries(steps=steps, raw=raw, smoothed=smoothed, wall_times=wall_times)


def compute_steps_per_sec(series: LossSeries, *, window: int = 10) -> Optional[float]:
    """Average it/s over the last ``window`` step deltas in the series.

    Returns ``None`` when:
      - we have fewer than 2 samples (no delta to measure)
      - all samples in the window share the same wall-clock (would divide by zero)
      - wall_times wasn't populated by the source

    Computed from tfevents wall_time, not request time, so the result is
    accurate even when the snapshot is built well after the trainer wrote
    its last batch.
    """

    wts = series.wall_times
    steps = series.steps
    if not wts or len(wts) < 2 or len(wts) != len(steps):
        return None
    n = min(window, len(wts) - 1)
    # Last (n+1) samples → n deltas.
    dt = wts[-1] - wts[-1 - n]
    dsteps = steps[-1] - steps[-1 - n]
    if dt <= 0 or dsteps <= 0:
        return None
    return float(dsteps) / float(dt)


def parse_max_steps_from_log(run_dir: Path) -> Optional[int]:
    """Read the run's stdout.log for `--max_train_steps N`."""
    log = run_dir / "logs" / "stdout.log"
    if not log.exists():
        return None
    try:
        head = log.read_text(encoding="utf-8", errors="replace")[:4000]
    except OSError:
        return None
    # Look for "--max_train_steps N" or "'--max_train_steps', 'N'"
    import re
    m = re.search(r"--max_train_steps['\"\s,]+(\d+)", head)
    if m:
        return int(m.group(1))
    return None


_TRAINING_ROLES = ("baseline", "candidate", "curated", "finalist")


def score_history_rows(ledger_rows: list[dict]) -> list[CandidateRow]:
    out: list[CandidateRow] = []
    for r in ledger_rows:
        if r.get("role") not in _TRAINING_ROLES:
            continue
        comps = r.get("score_components") or {}
        score = r.get("score")
        if isinstance(score, str):  # serialized inf/nan
            score = None
        out.append(CandidateRow(
            run_id=r.get("run_id", ""),
            role=r.get("role", ""),
            config_id=r.get("config_id", ""),
            score=score,
            final_smoothed=comps.get("final_smoothed"),
            slope=comps.get("slope"),
            n_steps=int(r.get("n_steps", 0) or 0),
            duration_s=float(r.get("duration_s", 0.0) or 0.0),
            disqualified=r.get("disqualified"),
        ))
    return out


def setup_status(ledger_rows: list[dict]) -> str:
    setup_rows = [r for r in ledger_rows if r.get("role") == "setup"]
    if not setup_rows:
        return "not started"
    if all(r.get("exit_code") == 0 for r in setup_rows):
        return "done"
    if any(r.get("exit_code") in (None, 0) for r in setup_rows):
        return "running"
    return "errored"


def _summarise_judge_failures(rows_with_judge: list[dict]) -> str:
    """Human-readable breakdown of WHY judgements failed.

    Bracket records a per-image error string on every failed judgement
    (``judge_report.judgements[i].error``). This bins them into a few
    actionable categories so the user gets one suggestion line instead
    of staring at "(N failed)" with no context.
    """

    buckets: dict[str, int] = {}
    for r in rows_with_judge:
        for j in (r.get("judge_report") or {}).get("judgements") or []:
            err = j.get("error")
            if not err:
                continue
            err_s = str(err)
            if "no JSON scores" in err_s:
                buckets["model returned prose, not JSON"] = buckets.get(
                    "model returned prose, not JSON", 0,
                ) + 1
            elif "non-JSON response" in err_s or "empty choices" in err_s:
                buckets["LMStudio response malformed"] = buckets.get(
                    "LMStudio response malformed", 0,
                ) + 1
            elif "URLError" in err_s or "timeout" in err_s.lower() or "ConnectionError" in err_s:
                buckets["LMStudio unreachable"] = buckets.get(
                    "LMStudio unreachable", 0,
                ) + 1
            else:
                buckets["other"] = buckets.get("other", 0) + 1
    if not buckets:
        return ""
    parts = ", ".join(f"{n} {label}" for label, n in sorted(buckets.items(), key=lambda kv: -kv[1]))
    hint = ""
    if buckets.get("model returned prose, not JSON"):
        hint = (
            " — try a non-thinking VLM (Qwen2.5-VL-7B, MiniCPM-V) or raise "
            "max_tokens; LMStudio's structured-output mode (json_schema) is "
            "already enabled by default."
        )
    elif buckets.get("LMStudio unreachable"):
        hint = " — confirm LMStudio is running and the configured base_url + port are reachable."
    return f"   ↳ failure breakdown: {parts}.{hint}"


def build_snapshot(
    output_dir: Path,
    *,
    total_runs_target: int = 0,
    ema_alpha: float = 0.05,
    judge_configured: bool = False,
) -> MonitorSnapshot:
    output_dir = Path(output_dir)
    ledger_path = output_dir / "ledger.jsonl"
    rows = read_ledger(ledger_path)
    history = score_history_rows(rows)
    completed = sum(1 for r in rows if r.get("role") in _TRAINING_ROLES)
    progress = (completed / total_runs_target * 100.0) if total_runs_target else 0.0
    progress = min(100.0, progress)

    current_run = latest_running_run(output_dir, rows)
    cur_steps_done: Optional[int] = None
    cur_max_steps: Optional[int] = None
    cur_loss: Optional[LossSeries] = None
    cur_run_id: Optional[str] = None
    cur_steps_per_sec: Optional[float] = None
    if current_run is not None:
        cur_run_id = current_run.name
        cur_max_steps = parse_max_steps_from_log(current_run)
        tfe = find_tfevents_in(current_run)
        if tfe is not None:
            cur_loss = load_loss_series(tfe, ema_alpha=ema_alpha)
            if cur_loss is not None and cur_loss.steps:
                cur_steps_done = cur_loss.steps[-1]
                cur_steps_per_sec = compute_steps_per_sec(cur_loss)

    setup = setup_status(rows)
    if setup == "running":
        status_line = "**Setup**: pre-caching latents + text-encoder outputs (one-time)."
    elif current_run is not None:
        status_line = (f"**Running** {cur_run_id}"
                       + (f" · step {cur_steps_done}/{cur_max_steps}" if cur_steps_done and cur_max_steps else "")
                       + f" · {completed}/{total_runs_target} runs done"
                       if total_runs_target else "")
    elif completed > 0 and (not total_runs_target or completed >= total_runs_target):
        status_line = f"**Done.** {completed} runs recorded."
    elif setup == "errored":
        status_line = "**Setup errored** — check the latest setup-*.log under runs/setup/."
    else:
        status_line = "_idle / no session yet_"

    # Judge state. We have two signals:
    #   1. INTENT — `judge_configured`, which the API passes through from
    #      the start-session request. True means the user picked LMStudio +
    #      gave a sample-prompts file.
    #   2. EVIDENCE — at least one ledger row has a populated `judge_report`.
    #      Means the judge has actually scored a sample.
    # Show the right message for the cross-product so a user who configured
    # the judge but stopped before any run finished doesn't see a misleading
    # "not configured" warning.
    training_rows = [r for r in rows if r.get("role") in _TRAINING_ROLES]
    rows_with_judge = [r for r in training_rows if r.get("judge_report")]
    if not training_rows and not judge_configured:
        judge_summary = ""
    elif rows_with_judge:
        n_imgs = sum(int(r["judge_report"].get("n_images", 0)) for r in rows_with_judge)
        n_failed = sum(int(r["judge_report"].get("n_failed", 0)) for r in rows_with_judge)
        means = [float(r["judge_report"].get("mean_overall", 0.0)) for r in rows_with_judge]
        avg = sum(means) / len(means) if means else 0.0
        judge_summary = (
            f"**Judge:** ✓ LMStudio · {len(rows_with_judge)}/{len(training_rows)} runs scored · "
            f"{n_imgs} images judged ({n_failed} failed) · mean visual score **{avg:.2f}/10** "
            f"(over {n_imgs - n_failed} successful)"
        )
        if n_failed:
            judge_summary += "\n" + _summarise_judge_failures(rows_with_judge)
    elif judge_configured:
        # Configured but no scored row yet — distinct from "not configured".
        judge_summary = (
            "**Judge:** ✓ LMStudio configured — waiting for the first run to "
            "complete so samples can be scored."
        )
    else:
        judge_summary = (
            "**Judge:** ❌ not configured — runs scored on training loss only. "
            "Enable LMStudio + sample_prompts on the Setup tab to score sample images."
        )

    # session_done: training target reached AND no run currently in-flight.
    session_done = (
        total_runs_target > 0 and completed >= total_runs_target
        and current_run is None
    )

    # Count rows the divergence killer terminated. The runner records
    # ``error="pruned:..."`` on those rows; we surface a small counter
    # so the Monitor tab can show "killed: K / running: R / completed: C".
    killed_by_pruner = sum(
        1 for r in rows
        if isinstance(r.get("error"), str) and r["error"].startswith("pruned")
    )
    running_runs = 1 if current_run is not None else 0

    return MonitorSnapshot(
        status_line=status_line,
        progress_pct=progress,
        completed_runs=completed,
        total_runs_target=total_runs_target,
        current_run_id=cur_run_id,
        current_run_steps_done=cur_steps_done,
        current_run_max_steps=cur_max_steps,
        current_loss=cur_loss,
        score_history=history,
        setup_status=setup,
        judge_summary=judge_summary,
        session_done=session_done,
        current_steps_per_sec=cur_steps_per_sec,
        killed_by_pruner=killed_by_pruner,
        running_runs=running_runs,
    )


_SAMPLE_ASSET_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp4", ".webm", ".mov", ".mkv",
)


def _iter_sample_assets(sample_dir: Path) -> Iterable[Path]:
    """Yield image and video sample files from `sample_dir`. Skips the
    `_frames/` subdir produced by the scorer's video frame extractor."""
    for entry in sample_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.suffix.lower() in _SAMPLE_ASSET_EXTS:
            yield entry


def gallery_items(output_dir: Path, max_items: int = 200) -> list[tuple[str, str]]:
    """Return (image_path, caption) pairs from every run's samples/ dir.

    Flat list — kept for tests and the Results tab. The Monitor tab uses
    `gallery_groups()` instead so it can render one collapsible accordion
    per run.
    """
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return []
    out: list[tuple[str, str]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        sample_dir = run_dir / "output" / "sample"
        if not sample_dir.exists():
            continue
        for asset in sorted(_iter_sample_assets(sample_dir)):
            out.append((str(asset), f"{run_dir.name} / {asset.name}"))
            if len(out) >= max_items:
                return out
    return out


@dataclass
class GalleryGroup:
    run_id: str
    items: list[tuple[str, str]]   # (image_path, caption) pairs
    mtime: float                   # newest image mtime in this group


def gallery_groups(
    output_dir: Path, *, max_items_per_run: int = 200,
) -> list[GalleryGroup]:
    """Return one group per run that has samples, ordered newest-first.

    Within a group, images are sorted newest-first so the most recent
    sampling pass is what you see when an accordion opens.
    """
    output_dir = Path(output_dir)
    runs_dir = output_dir / "runs"
    if not runs_dir.exists():
        return []
    groups: list[GalleryGroup] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "setup":
            continue
        sample_dir = run_dir / "output" / "sample"
        if not sample_dir.exists():
            continue
        imgs = list(_iter_sample_assets(sample_dir))
        if not imgs:
            continue
        imgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        imgs = imgs[:max_items_per_run]
        items = [(str(p), p.name) for p in imgs]
        newest = max((p.stat().st_mtime for p in imgs), default=0.0)
        groups.append(GalleryGroup(run_id=run_dir.name, items=items, mtime=newest))
    groups.sort(key=lambda g: g.mtime, reverse=True)
    return groups
