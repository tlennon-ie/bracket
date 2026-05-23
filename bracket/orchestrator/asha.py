"""Two-rung ASHA wrapper around :func:`bracket.orchestrator.loop.orchestrate`.

Rationale: the orchestrator's existing flow runs every config to the full
``max_steps_per_run`` budget. For a search budget of 8-16 candidates and a
typical step budget where loss-curve shape is visible at ~25% of the run,
a two-rung schedule is cheap insurance: spend 25% on every config, promote
the top 50% to the full budget.

Why a wrapper and not a flag inside ``orchestrate``: the orchestrate
function is already long and the change is opt-in. Keeping it separate
also lets the rung-1 pass record its rows under a non-reporting role so
the proof report only sees the rung-2 entries.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from bracket.judge.base import SampleJudge
from bracket.orchestrator.ledger import Ledger
from bracket.orchestrator.loop import (
    OrchestrationResult,
    _config_id_from_history,
    config_id,
    orchestrate,
)
from bracket.orchestrator.pruner import PartialResult, promote_top_half
from bracket.search.controller import LedgerEntry, SearchController
from bracket.search.space import SearchOverrides
from bracket.trainer.base import Trainer

logger = logging.getLogger("bracket.orchestrator.asha")


# Trainers whose --resume semantics are known good for the two-rung gate.
# Everything else falls back to "restart from scratch at full budget".
_RESUME_SUPPORTED_TRAINERS: frozenset[str] = frozenset({
    # sd-scripts family (--resume <state_dir>):
    "sdxl-sd-scripts",
    "sdxl-full-sd-scripts",
    "sd35-lora-sd-scripts",
    "sd35-full-sd-scripts",
    # musubi-tuner family (similar):
    "flux1-lora-musubi",
    "flux1-full-musubi",
    "flux1-kontext-lora-musubi",
    "flux2-klein-lora-musubi",
    "zimage-lora-musubi",
    "zimage-full-musubi",
    "wan-lora-musubi",
    "wan-full-musubi",
    "hunyuan-video-lora-musubi",
    "hunyuan-video-full-musubi",
    "qwen-image-lora-musubi",
    "qwen-image-full-musubi",
    "qwen-image-edit-lora-musubi",
    "ltx-video-lora-musubi",
    "framepack-lora-musubi",
})


def trainer_supports_resume(trainer_name: str) -> bool:
    """True if ``--resume`` semantics are validated for the trainer.

    Unknown trainers default to False — the safe option. When False, the
    promotion gate falls back to ``promote = full budget restart``.
    """
    return trainer_name in _RESUME_SUPPORTED_TRAINERS


def rung1_max_steps(max_steps_per_run: int) -> int:
    """Compute the rung-1 step budget. 25% of full, floored at 25 steps so
    even tiny demos still produce a non-trivial loss curve."""
    return max(25, int(max_steps_per_run) // 4)


def select_promoted_from_rung1(rung1_history: list[LedgerEntry]) -> set[str]:
    """Translate a rung-1 LedgerEntry list to a set of promoted config_ids.

    The promotion gate uses :func:`promote_top_half`. Disqualified or
    erroring trials are filtered out before ranking — they never promote.
    """
    by_cfg: dict[str, list[LedgerEntry]] = {}
    for h in rung1_history:
        cid = _config_id_from_history(h)
        by_cfg.setdefault(cid, []).append(h)
    partials: list[PartialResult] = []
    for cid, group in by_cfg.items():
        scored = [g for g in group if g.score is not None]
        if not scored:
            continue
        mean = sum(g.score for g in scored) / len(scored)
        partials.append(PartialResult(
            run_id=cid,
            n_steps_completed=int(scored[-1].n_steps),
            smoothed_loss_tail=float(mean),
            grad_norm_p99=None,
            walltime_s=float(scored[-1].duration_s),
        ))
    return promote_top_half(partials)


def _rewrite_rung1_roles(ledger_path: Path) -> None:
    """Mark every ``baseline``/``curated``/``candidate`` row in this ledger as
    a rung-1 row so the proof report ignores them.

    The wrapper runs the rung-1 ledger in a sibling directory and then
    merges rows into the user's session ledger with the rung-1 marker on
    the role, so the orchestrator's resume guards remain happy.
    """
    if not ledger_path.exists():
        return
    rewritten: list[str] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            rewritten.append(line)
            continue
        role = row.get("role")
        if role in ("baseline", "candidate", "curated"):
            row["role"] = f"rung1-{role}"
        rewritten.append(json.dumps(row))
    ledger_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")


def orchestrate_two_rung(
    *,
    trainer: Trainer,
    dataset_toml: Path,
    output_dir: Path,
    controller: SearchController,
    budget_runs: int,
    max_steps_per_run: int,
    max_wall_seconds_per_run: int,
    base_seed: int = 42,
    seeds_per_config: int = 1,
    skip_baseline: bool = False,
    n_curated: Optional[int] = None,
    mirror_stdout: bool = False,
    sample_prompts: Optional[Path] = None,
    sample_every_n_steps: Optional[int] = None,
    sample_judge: Optional[SampleJudge] = None,
    loss_weight: float = 0.3,
    sample_weight: float = 0.7,
    stop_event=None,
    search_overrides: Optional[SearchOverrides] = None,
    enable_divergence_killer: bool = True,
    on_launcher_ready: Optional[Callable[[object], None]] = None,
) -> OrchestrationResult:
    """Run :func:`orchestrate` twice — rung 1 at 25% budget, rung 2 promoting
    the top half. The merged ledger lives at ``output_dir/ledger.jsonl``."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rung1_dir = output_dir / "_rung1"
    rung1_dir.mkdir(parents=True, exist_ok=True)

    rung1_steps = rung1_max_steps(max_steps_per_run)
    # We *want* rung-1 to leave state dirs behind so rung-2 can resume.
    # The trainer prepare_run() interface accepts save_state but it's not
    # threaded through the orchestrator. The pragmatic compromise: still
    # halve the per-run wall clock for rung 1 so very-slow trainers don't
    # blow the budget.
    rung1_wall = max(60, max_wall_seconds_per_run // 2)
    logger.info(
        "asha: starting rung 1 — %d trials at %d steps, wall=%ds",
        budget_runs, rung1_steps, rung1_wall,
    )
    rung1 = orchestrate(
        trainer=trainer,
        dataset_toml=dataset_toml,
        output_dir=rung1_dir,
        controller=controller,
        budget_runs=budget_runs,
        max_steps_per_run=rung1_steps,
        max_wall_seconds_per_run=rung1_wall,
        base_seed=base_seed,
        seeds_per_config=seeds_per_config,
        skip_baseline=skip_baseline,
        n_curated=n_curated,
        mirror_stdout=mirror_stdout,
        sample_prompts=sample_prompts,
        sample_every_n_steps=sample_every_n_steps,
        sample_judge=sample_judge,
        loss_weight=loss_weight,
        sample_weight=sample_weight,
        stop_event=stop_event,
        search_overrides=search_overrides,
        enable_divergence_killer=enable_divergence_killer,
        enable_two_rung_asha=False,  # never recursive
        on_launcher_ready=on_launcher_ready,
    )

    # Decide who gets promoted.
    promoted_cids = select_promoted_from_rung1(rung1.history)
    logger.info(
        "asha: rung 1 done — %d configs trained, %d promoted to rung 2",
        len({_config_id_from_history(h) for h in rung1.history}),
        len(promoted_cids),
    )

    # Re-tag rung-1 rows then copy into the merged ledger.
    rung1_ledger_path = rung1_dir / "ledger.jsonl"
    _rewrite_rung1_roles(rung1_ledger_path)
    merged_ledger = Ledger(output_dir / "ledger.jsonl")
    if rung1_ledger_path.exists():
        for line in rung1_ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                merged_ledger.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not promoted_cids:
        logger.warning("asha: nothing to promote — returning rung-1 best as final.")
        return rung1

    # Rung 2: rerun the promoted configs at the full budget on the user's
    # output_dir, so the canonical ledger captures the authoritative scores.
    promoted_configs = []
    for cid in promoted_cids:
        match = next(
            (h for h in rung1.history if _config_id_from_history(h) == cid),
            None,
        )
        if match is None:
            continue
        promoted_configs.append((cid, dict(match.config)))

    rung2_dir = output_dir / "_rung2"
    rung2_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "asha: starting rung 2 — %d promoted configs at %d steps",
        len(promoted_configs), max_steps_per_run,
    )
    rung2_result = _run_rung2_pass(
        trainer=trainer,
        dataset_toml=dataset_toml,
        output_dir=rung2_dir,
        promoted_configs=promoted_configs,
        max_steps=max_steps_per_run,
        max_wall_seconds=max_wall_seconds_per_run,
        base_seed=base_seed,
        seeds_per_config=seeds_per_config,
        mirror_stdout=mirror_stdout,
        sample_prompts=sample_prompts,
        sample_every_n_steps=sample_every_n_steps,
        sample_judge=sample_judge,
        loss_weight=loss_weight,
        sample_weight=sample_weight,
        stop_event=stop_event,
        on_launcher_ready=on_launcher_ready,
    )
    # Copy rung-2 ledger rows into the merged session ledger.
    rung2_ledger = rung2_dir / "ledger.jsonl"
    if rung2_ledger.exists():
        for line in rung2_ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                merged_ledger.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    history = list(merged_ledger.to_history())
    # "Best" is whichever rung-2 entry got the lowest score.
    rung2_history = [h for h in history if h.run_id.startswith("rung2-")]
    scored = [h for h in rung2_history if h.score is not None]
    best = min(scored, key=lambda h: h.score) if scored else None
    baseline = next((h for h in history if h.run_id.startswith("baseline-")), None)
    n_disqualified = sum(1 for h in history if h.score is None)
    return OrchestrationResult(
        session_dir=output_dir,
        ledger_path=merged_ledger.path,
        history=history,
        baseline_entry=baseline,
        best_entry=best,
        n_completed=len(history),
        n_disqualified=n_disqualified,
    )


def _run_rung2_pass(
    *,
    trainer: Trainer,
    dataset_toml: Path,
    output_dir: Path,
    promoted_configs: list[tuple[str, dict]],
    max_steps: int,
    max_wall_seconds: int,
    base_seed: int,
    seeds_per_config: int,
    mirror_stdout: bool,
    sample_prompts: Optional[Path],
    sample_every_n_steps: Optional[int],
    sample_judge: Optional[SampleJudge],
    loss_weight: float,
    sample_weight: float,
    stop_event,
    on_launcher_ready: Optional[Callable[[object], None]] = None,
) -> None:
    """Run each promoted config at full budget. Resume from rung-1 state is
    not currently threaded through to the trainer — the gate falls back to
    a from-scratch restart for now. See ``orchestrate_two_rung`` docstring."""
    from bracket.judge.base import parse_judge_prompts_file
    from bracket.orchestrator.loop import _execute_one, _record
    from bracket.orchestrator.runner import RunLauncher
    from bracket.orchestrator.scorer import Scorer

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output_dir / "ledger.jsonl")
    launcher = RunLauncher(
        max_wall_seconds=max_wall_seconds,
        mirror_stdout=mirror_stdout,
        stop_event=stop_event,
    )
    if on_launcher_ready is not None:
        try:
            on_launcher_ready(launcher)
        except Exception:  # noqa: BLE001
            logger.exception("on_launcher_ready callback raised")
    scorer = Scorer(
        sample_judge=sample_judge if sample_prompts is not None else None,
        loss_weight=loss_weight,
        sample_weight=sample_weight,
    )
    judge_prompts = parse_judge_prompts_file(sample_prompts) if sample_prompts else None

    space = trainer.declare_search_space()
    for cand_idx, (cid, cfg_dict) in enumerate(promoted_configs):
        if stop_event is not None and stop_event.is_set():
            logger.info("asha: stop requested during rung 2")
            return
        config = trainer.config_from_dict(cfg_dict)
        for seed_idx in range(seeds_per_config):
            seed = base_seed + (cand_idx + 1) * 100 + seed_idx
            run_id = f"rung2-{cand_idx:03d}-s{seed_idx}-{int(time.time())}"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            result, score, _live = _execute_one(
                trainer=trainer, config=config, dataset_toml=dataset_toml,
                max_steps=max_steps, seed=seed, run_dir=run_dir,
                launcher=launcher, scorer=scorer, run_id=run_id,
                sample_prompts=sample_prompts,
                sample_every_n_steps=sample_every_n_steps,
                judge_prompts=judge_prompts,
            )
            _record(
                ledger, run_id=run_id, role="candidate", cfg_id=cid,
                config_dict=cfg_dict, seed=seed, seed_idx=seed_idx,
                result=result, score=score,
                trainer_name=trainer.name, search_space_name=space.name,
            )
