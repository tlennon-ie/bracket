"""Promote a search candidate to a full training run.

The promote flow takes one ledger row from a finished (or in-progress)
search session and runs its hyperparameters through ``_execute_one``
against the **full** dataset_toml — not the search-time subset — with
a longer step budget and intermediate checkpoint saves.

Conceptually a "finals stage of one", but with:
  - No multi-seed (the user already picked the winner).
  - Explicit save_every_n_steps + save_state so the run produces
    resumable checkpoints, not just a single end-of-run save.
  - Optional ``resume_from`` to warm-start from a prior LoRA or state.
  - Role recorded as ``"promoted"`` so the UI / report can distinguish
    promoted full runs from search-stage runs.

This is the v0.x entry point — it's a single function call from the
API layer, not a long-lived orchestrator object. If the user wants a
multi-stage promote pipeline (e.g. fine-tune then merge LoRAs) that's
a future iteration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bracket.judge.base import SampleJudge, parse_judge_prompts_file
from bracket.orchestrator.ledger import Ledger
from bracket.orchestrator.loop import (
    OrchestrationResult,
    _execute_one,
    _record,
    config_id,
)
from bracket.orchestrator.runner import RunLauncher
from bracket.orchestrator.scorer import Scorer
from bracket.search.controller import LedgerEntry
from bracket.trainer.base import Trainer

logger = logging.getLogger("bracket.orchestrator.promote")


@dataclass
class PromoteResult:
    """What ``run_promoted`` returns to the caller."""

    promoted_run_id: str
    run_dir: Path
    entry: Optional[LedgerEntry]
    duration_s: float


def run_promoted(
    *,
    trainer: Trainer,
    source_config: dict[str, object],
    dataset_toml: Path,
    output_dir: Path,
    max_steps: int,
    seed: int = 42,
    save_every_n_steps: int = 500,
    save_state: bool = True,
    resume_from: Optional[Path] = None,
    sample_prompts: Optional[Path] = None,
    sample_every_n_steps: Optional[int] = None,
    sample_judge: Optional[SampleJudge] = None,
    loss_weight: float = 0.3,
    sample_weight: float = 0.7,
    max_wall_seconds: int = 14 * 24 * 3600,  # one week — effectively unbounded
    base_run_id: str = "",
) -> PromoteResult:
    """Run a single training pass with a candidate's hyperparameters at
    full scale. Appends one row to the same ``ledger.jsonl`` so the proof
    report covers it.

    Parameters
    ----------
    trainer
        The trainer adapter to drive (must match the original search-run
        family — caller is responsible for matching).
    source_config
        The ledger row's ``config`` dict for the winning candidate. Will
        be re-validated by ``trainer.config_from_dict`` before launch.
    dataset_toml
        Path to the FULL dataset TOML (not the subset). Promote-flow
        explicitly does not call ``build_subset`` again.
    output_dir
        Session output directory (the same one the search ran in). The
        promoted run lands under ``<output_dir>/runs/<promoted_run_id>``.
    max_steps
        Step budget for the full run. 2000 is a reasonable default.
    save_every_n_steps
        Cadence for intermediate checkpoint saves. Combined with
        ``save_state=True`` this produces resumable snapshots.
    sample_every_n_steps
        Sample-image cadence. When None and ``sample_prompts`` is given,
        defaults to ``save_every_n_steps`` so each checkpoint comes with
        a sample sheet.
    resume_from
        Optional path to a prior LoRA or training-state directory to
        warm-start from.
    """

    output_dir = Path(output_dir).resolve()
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output_dir / "ledger.jsonl")
    launcher = RunLauncher(max_wall_seconds=max_wall_seconds)
    scorer = Scorer(
        sample_judge=sample_judge if sample_prompts is not None else None,
        loss_weight=loss_weight, sample_weight=sample_weight,
    )
    judge_prompts = parse_judge_prompts_file(sample_prompts) if sample_prompts else None

    config = trainer.config_from_dict(dict(source_config))
    cid = config_id(dict(source_config))
    timestamp = int(time.time())
    if base_run_id:
        run_id = f"promoted-{base_run_id[:16]}-{timestamp}"
    else:
        run_id = f"promoted-{cid[:8]}-{timestamp}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # If sampling is requested but cadence wasn't set, align it with the
    # checkpoint cadence so each saved snapshot ships a matching sample
    # sheet.
    effective_sample_every = sample_every_n_steps
    if sample_prompts is not None and effective_sample_every is None:
        effective_sample_every = save_every_n_steps

    logger.info(
        "promote: run_id=%s cfg=%s max_steps=%d save_every=%d save_state=%s resume_from=%s",
        run_id, cid, max_steps, save_every_n_steps, save_state, resume_from,
    )
    start = time.monotonic()
    result, score, _live = _execute_one(
        trainer=trainer, config=config, dataset_toml=dataset_toml,
        max_steps=max_steps, seed=seed, run_dir=run_dir,
        launcher=launcher, scorer=scorer, run_id=run_id,
        sample_prompts=sample_prompts,
        sample_every_n_steps=effective_sample_every,
        judge_prompts=judge_prompts,
        save_every_n_steps=save_every_n_steps,
        save_state=save_state,
        resume_from=resume_from,
    )
    duration_s = time.monotonic() - start
    entry = _record(
        ledger, run_id=run_id, role="promoted", cfg_id=cid,
        config_dict=dict(source_config), seed=seed, seed_idx=0,
        result=result, score=score,
        trainer_name=type(trainer).__name__,
    )
    logger.info(
        "promote: done run_id=%s score=%s dq=%s duration=%.1fs",
        run_id, score.score, score.disqualified, duration_s,
    )
    return PromoteResult(
        promoted_run_id=run_id,
        run_dir=run_dir,
        entry=entry,
        duration_s=duration_s,
    )


__all__ = ["PromoteResult", "run_promoted"]


# Help downstream consumers (and the OrchestrationSession's threading
# wrapper) by exporting an `OrchestrationResult`-shaped wrapper so the
# session.start() callable contract is unchanged.
def run_promoted_for_session(
    *,
    trainer: Trainer,
    source_config: dict[str, object],
    dataset_toml: Path,
    output_dir: Path,
    max_steps: int,
    save_every_n_steps: int,
    save_state: bool,
    resume_from: Optional[Path],
    sample_prompts: Optional[Path],
    sample_every_n_steps: Optional[int],
    sample_judge: Optional[SampleJudge],
    loss_weight: float,
    sample_weight: float,
    base_run_id: str,
) -> OrchestrationResult:
    """Adapter for :class:`OrchestrationSession`.

    ``OrchestrationSession.start`` accepts a callable returning
    :class:`OrchestrationResult`. We synthesise one from the single
    promoted run so the session machinery (status, snapshots, stop)
    works without modification.
    """

    result = run_promoted(
        trainer=trainer, source_config=source_config,
        dataset_toml=dataset_toml, output_dir=output_dir,
        max_steps=max_steps,
        save_every_n_steps=save_every_n_steps,
        save_state=save_state, resume_from=resume_from,
        sample_prompts=sample_prompts,
        sample_every_n_steps=sample_every_n_steps,
        sample_judge=sample_judge,
        loss_weight=loss_weight, sample_weight=sample_weight,
        base_run_id=base_run_id,
    )
    history = [result.entry] if result.entry is not None else []
    best = (
        result.entry
        if (result.entry is not None and result.entry.score is not None)
        else None
    )
    return OrchestrationResult(
        history=history, best_entry=best, baseline_entry=None,
    )
