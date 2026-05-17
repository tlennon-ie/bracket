"""Autonomous loop with multi-seed support.

For each candidate config, we may run it `seeds_per_config` times with
different seeds. The ledger records one row per (config, seed). At report
time, rows are grouped by `config_id` (a deterministic hash of the config
dict) and aggregated for ranking and confidence statistics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from bracket.dataset.validator import validate_dataset_toml
from bracket.judge.base import SampleJudge, parse_judge_prompts_file
from bracket.orchestrator.history import (
    lookup_warmstart_configs,
    record_session_winner,
)
from bracket.orchestrator.ledger import Ledger
from bracket.orchestrator.runner import RunLauncher, RunResult
from bracket.orchestrator.scorer import Scorer, ScoreReport
from bracket.search.controller import LedgerEntry, SearchController
from bracket.search.space import (
    SearchOverrides,
    apply_search_overrides,
    clamp_config_to_overrides,
)
from bracket.trainer.base import Trainer, TrainerConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    session_dir: Path
    ledger_path: Path
    history: list[LedgerEntry]
    baseline_entry: Optional[LedgerEntry]
    best_entry: Optional[LedgerEntry]
    n_completed: int
    n_disqualified: int


def config_id(config_dict: dict) -> str:
    """Deterministic short hash of a config dict — groups multi-seed runs."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _make_run_id(prefix: str, candidate_idx: int, seed_idx: int) -> str:
    return f"{prefix}-{candidate_idx:03d}-s{seed_idx}-{int(time.time())}"


def _execute_one(
    *,
    trainer: Trainer,
    config: TrainerConfig,
    dataset_toml: Path,
    max_steps: int,
    seed: int,
    run_dir: Path,
    launcher: RunLauncher,
    scorer: Scorer,
    run_id: str,
    sample_prompts: Optional[Path] = None,
    sample_every_n_steps: Optional[int] = None,
    judge_prompts: Optional[list[str]] = None,
    n_in_dist_prompts: Optional[int] = None,
    # Promote-flow extensions (default to current search-run behaviour).
    save_every_n_steps: Optional[int] = None,
    save_state: bool = False,
    resume_from: Optional[Path] = None,
) -> tuple[RunResult, ScoreReport]:
    spec = trainer.prepare_run(
        run_dir=run_dir,
        config=config,
        dataset_toml=dataset_toml,
        max_steps=max_steps,
        seed=seed,
        sample_prompts=sample_prompts,
        sample_every_n_steps=sample_every_n_steps,
        save_every_n_steps=save_every_n_steps,
        save_state=save_state,
        resume_from=resume_from,
    )
    log_path = run_dir / "logs" / "stdout.log"
    result = launcher.launch(run_id, spec, log_path)
    score = scorer.score_run(
        tfevents_path=result.tfevents_path,
        sample_dir=spec.sample_dir,
        prompts=judge_prompts,
        n_in_dist_prompts=n_in_dist_prompts,
    )
    # Free the VLM's VRAM before the next training run starts. The judge's
    # eject() is a no-op when it has nothing to release; LMStudio's
    # implementation issues a best-effort unload.
    if scorer.sample_judge is not None:
        try:
            scorer.sample_judge.eject()
        except Exception:  # noqa: BLE001 — eject must never block the loop
            pass
    return result, score


def _record(
    ledger: Ledger,
    *,
    run_id: str,
    role: str,
    cfg_id: str,
    config_dict: dict,
    seed: int,
    seed_idx: int,
    result: RunResult,
    score: ScoreReport,
    trainer_name: str = "",
    search_space_name: str = "",
) -> LedgerEntry:
    score_value = None if score.score == float("inf") else score.score
    judge_payload = None
    if score.judge_report is not None:
        judgements_payload = []
        for j in score.judge_report.judgements:
            judgements_payload.append({
                "image": j.image_path.name,
                "prompt": j.prompt,
                "prompt_adherence": j.prompt_adherence,
                "visual_quality": j.visual_quality,
                "artifact_free": j.artifact_free,
                "overall": j.overall,
                "error": j.error,
                # Truncate raw_response — full responses can be multi-KB and
                # bloat the ledger. Keep enough context for debugging.
                "raw_response": (j.raw_response[:600] if j.raw_response else ""),
            })
        judge_payload = {
            "n_images": score.judge_report.n_images,
            "n_failed": score.judge_report.n_failed,
            "mean_overall": score.judge_report.mean_overall,
            "mean_prompt_adherence": score.judge_report.mean_prompt_adherence,
            "mean_visual_quality": score.judge_report.mean_visual_quality,
            "mean_artifact_free": score.judge_report.mean_artifact_free,
            "judgements": judgements_payload,
        }
    row = {
        "run_id": run_id,
        "role": role,
        "trainer_name": trainer_name,
        "search_space_name": search_space_name,
        "config_id": cfg_id,
        "config": config_dict,
        "seed": seed,
        "seed_idx": seed_idx,
        "score": score_value,
        "score_components": score.components,
        "judge_report": judge_payload,
        "n_steps": score.n_steps,
        "duration_s": result.duration_s,
        "exit_code": result.exit_code,
        "killed_by_timeout": result.killed_by_timeout,
        "error": result.error,
        "disqualified": score.disqualified,
        "tfevents_path": str(result.tfevents_path) if result.tfevents_path else None,
        "log_path": str(result.log_path),
        "sample_dir": str(result.spec.sample_dir) if result.spec.sample_dir else None,
        "timestamp": Ledger.now_ts(),
    }
    ledger.append(row)
    return LedgerEntry(
        run_id=run_id,
        config=config_dict,
        score=score_value,
        error=result.error,
        duration_s=result.duration_s,
        n_steps=score.n_steps,
        timestamp=row["timestamp"],
    )


def orchestrate(
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
    sample_prompts_ood: Optional[Path] = None,
    sample_every_n_steps: Optional[int] = None,
    sample_judge: Optional[SampleJudge] = None,
    loss_weight: float = 0.3,
    sample_weight: float = 0.7,
    stop_event: Optional["__import__('threading').Event"] = None,
    search_overrides: Optional[SearchOverrides] = None,
    use_history_priors: bool = False,
    history_db_path: Optional[Path] = None,
) -> OrchestrationResult:
    """`n_curated` controls the warm-start: number of curated configs to try
    after the baseline and before the search controller takes over. None or
    negative = run all configs the trainer publishes; 0 = skip warm-start
    entirely (legacy behaviour); positive int = cap. Each curated config
    counts as one candidate against `budget_runs`."""
    output_dir = Path(output_dir).resolve()

    validation_result = validate_dataset_toml(dataset_toml)
    if not validation_result.is_valid:
        logger.error("Dataset validation failed: %s", validation_result.summary)
        for err in validation_result.errors:
            if err.severity == "error":
                logger.error("  [ERROR] %s: %s", err.image_path, err.issue)
            else:
                logger.warning("  [WARNING] %s: %s", err.image_path, err.issue)
        raise RuntimeError(
            f"Dataset validation failed. See errors above. "
            f"You can:\n"
            f"  1. Fix captions manually\n"
            f"  2. Run: python -m bracket.dataset.validator --fix {dataset_toml}\n"
            f"  3. Run: python -m bracket.dataset.validator --remove-bad {dataset_toml}"
        )
    if validation_result.errors:
        logger.warning("Dataset has %d warning(s) but is trainable: %s",
                       len(validation_result.errors), validation_result.summary)
    else:
        logger.info("Dataset validation passed.")

    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output_dir / "ledger.jsonl")
    launcher = RunLauncher(
        max_wall_seconds=max_wall_seconds_per_run,
        mirror_stdout=mirror_stdout,
        stop_event=stop_event,
    )
    scorer = Scorer(
        sample_judge=sample_judge if sample_prompts is not None else None,
        loss_weight=loss_weight,
        sample_weight=sample_weight,
    )
    space = trainer.declare_search_space()
    space = apply_search_overrides(space, search_overrides)

    # When the user supplies a second (out-of-distribution) prompt file,
    # we concatenate it onto the in-distribution prompts and hand the
    # trainer a single merged file. Neither sd-scripts nor musubi-tuner
    # natively supports two prompt sets, and concatenation avoids paying
    # for a second sampling pass per checkpoint. The scorer is told the
    # in-dist count so it can split the resulting judgements back apart.
    n_in_dist_prompts: Optional[int] = None
    effective_sample_prompts: Optional[Path] = sample_prompts
    if sample_prompts is not None and sample_prompts_ood is not None:
        in_dist_prompts = parse_judge_prompts_file(sample_prompts)
        merged_path = output_dir / "merged_sample_prompts.txt"
        in_dist_raw = Path(sample_prompts).read_text(encoding="utf-8")
        ood_raw = Path(sample_prompts_ood).read_text(encoding="utf-8")
        if not in_dist_raw.endswith("\n"):
            in_dist_raw += "\n"
        merged_path.write_text(
            in_dist_raw + "# --- OOD prompts (held out) ---\n" + ood_raw,
            encoding="utf-8",
        )
        effective_sample_prompts = merged_path
        n_in_dist_prompts = len(in_dist_prompts)
        logger.info(
            "OOD prompts active: %d in-dist + %d OOD = %d total",
            n_in_dist_prompts,
            len(parse_judge_prompts_file(sample_prompts_ood)),
            len(parse_judge_prompts_file(merged_path)),
        )
    judge_prompts = (
        parse_judge_prompts_file(effective_sample_prompts)
        if effective_sample_prompts
        else None
    )

    history: list[LedgerEntry] = list(ledger.to_history())

    # Resume guard: if a prior session in this output_dir used a different
    # trainer or search-space, candidate scores aren't comparable across
    # sessions. Refuse to resume rather than silently produce mixed garbage.
    prior_trainer_names = {row.get("trainer_name") for row in ledger.iter_rows()}
    prior_trainer_names.discard(None)
    if prior_trainer_names and trainer.name not in prior_trainer_names:
        raise RuntimeError(
            f"trainer mismatch on resume: ledger has rows from "
            f"{sorted(prior_trainer_names)!r}, but this session uses "
            f"{trainer.name!r}. Use a fresh --output-dir."
        )
    prior_space_names = {row.get("search_space_name") for row in ledger.iter_rows()}
    prior_space_names.discard(None)
    if prior_space_names and space.name not in prior_space_names:
        raise RuntimeError(
            f"search-space mismatch on resume: ledger has rows from "
            f"{sorted(prior_space_names)!r}, but this session uses "
            f"{space.name!r}. Use a fresh --output-dir."
        )

    # Run any one-shot session setup the trainer needs (e.g. musubi latent
    # caching). Recorded as ledger rows with role="setup" so resume skips them.
    if not _has_completed_setup(ledger):
        setup_specs = trainer.session_setup_commands(
            dataset_toml=dataset_toml, run_dir=runs_dir / "setup",
        )
        for i, spec in enumerate(setup_specs):
            sid = f"setup-{i:03d}-{int(time.time())}"
            log_path = runs_dir / "setup" / f"{sid}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("session setup [%d/%d]: %s", i + 1, len(setup_specs),
                        Path(spec.cmd[-1] if not spec.cmd[-1].endswith('.py') else spec.cmd[2]).name
                        if spec.cmd else "?")
            result = launcher.launch(sid, spec, log_path)
            ledger.append({
                "run_id": sid,
                "role": "setup",
                "trainer_name": trainer.name,
                "search_space_name": space.name,
                "cmd": list(spec.cmd),
                "exit_code": result.exit_code,
                "duration_s": result.duration_s,
                "killed_by_timeout": result.killed_by_timeout,
                "error": result.error,
                "log_path": str(result.log_path),
                "timestamp": Ledger.now_ts(),
            })
            if result.exit_code not in (0, None):
                raise RuntimeError(
                    f"session setup step {sid} failed with exit code {result.exit_code}; "
                    f"see {result.log_path}"
                )

    # Decide whether the baseline has already been recorded.
    seen_config_ids = {
        _config_id_from_history(h) for h in history if h.run_id.startswith("baseline-")
    }
    baseline_entry: Optional[LedgerEntry] = next(
        (h for h in history if h.run_id.startswith("baseline-")), None
    )
    n_disqualified = sum(1 for h in history if h.score is None)

    if baseline_entry is None and not skip_baseline:
        # Honour the user's lr/batch_size bounds on the fixed baseline too —
        # otherwise a user who sets batch_size_min=6 still gets a baseline
        # at the trainer's tier default (e.g. 2) which contradicts the
        # explicit lower bound.
        config = clamp_config_to_overrides(trainer.baseline_config(), search_overrides)
        config_dict = config.to_dict()
        cfg_id = config_id(config_dict)
        for seed_idx in range(seeds_per_config):
            seed = base_seed + seed_idx
            run_id = _make_run_id("baseline", 0, seed_idx)
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info("running baseline %s seed=%d", run_id, seed)
            result, score = _execute_one(
                trainer=trainer, config=config, dataset_toml=dataset_toml,
                max_steps=max_steps_per_run, seed=seed, run_dir=run_dir,
                launcher=launcher, scorer=scorer, run_id=run_id,
                sample_prompts=effective_sample_prompts, sample_every_n_steps=sample_every_n_steps,
                judge_prompts=judge_prompts, n_in_dist_prompts=n_in_dist_prompts,
            )
            entry = _record(
                ledger, run_id=run_id, role="baseline", cfg_id=cfg_id,
                config_dict=config_dict, seed=seed, seed_idx=seed_idx,
                result=result, score=score,
                trainer_name=trainer.name, search_space_name=space.name,
            )
            if seed_idx == 0:
                baseline_entry = entry
            history.append(entry)
            if score.disqualified:
                n_disqualified += 1
            logger.info("baseline seed=%d score=%s dq=%s", seed_idx, score.score, score.disqualified)

    candidate_history = [h for h in history if not h.run_id.startswith("baseline-")]
    # Both "cur-" (curated warm-start) and "cand-" (controller-picked) trials
    # count toward budget. Resume picks up where the prior session left off.
    completed_curated = _count_unique_configs(history, role_prefix="cur-")
    completed_candidate_configs = _count_unique_configs(history, role_prefix="cand-")
    candidate_idx = completed_curated + completed_candidate_configs

    # Curated warm-start: run the trainer's published known-good configs
    # before handing off to the search controller. Each curated config
    # consumes `seeds_per_config` slots of budget_runs (which is in
    # seed-runs, matching controller.should_stop).
    curated_base = trainer.curated_configs()
    if use_history_priors:
        history_dicts = lookup_warmstart_configs(
            trainer.name, space.name, n=3, db_path=history_db_path,
        )
        existing_ids = {config_id(c.to_dict()) for c in curated_base}
        prepended: list[TrainerConfig] = []
        for cfg_dict in history_dicts:
            try:
                cfg = trainer.config_from_dict(cfg_dict)
            except Exception as e:  # noqa: BLE001 — old configs may have stale keys
                logger.warning(
                    "history: skipping warm-start config (incompatible with current trainer): %s",
                    e,
                )
                continue
            cid = config_id(cfg.to_dict())
            if cid in existing_ids:
                continue
            existing_ids.add(cid)
            prepended.append(cfg)
        if prepended:
            logger.info(
                "history: prepending %d warm-start config(s) for %s/%s",
                len(prepended), trainer.name, space.name,
            )
        curated_base = prepended + list(curated_base)
    curated = [
        clamp_config_to_overrides(c, search_overrides)
        for c in curated_base
    ]
    if n_curated is None or n_curated < 0:
        n_curated_target = len(curated)
    else:
        n_curated_target = min(int(n_curated), len(curated))
    remaining_budget_seedruns = max(0, budget_runs - len(candidate_history))
    max_curated_by_budget = remaining_budget_seedruns // max(1, seeds_per_config)
    n_curated_target = min(n_curated_target, max_curated_by_budget)

    completed_curated_ids = {
        _config_id_from_history(h) for h in history if h.run_id.startswith("cur-")
    }
    curated_to_run = [
        c for c in curated[:n_curated_target]
        if config_id(c.to_dict()) not in completed_curated_ids
    ]
    for cur_config in curated_to_run:
        if stop_event is not None and stop_event.is_set():
            logger.info("stop requested — exiting curated warm-start")
            break
        if len(candidate_history) >= budget_runs:
            break
        config_dict = cur_config.to_dict()
        cfg_id = config_id(config_dict)
        logger.info("running curated %d/%d cfg_id=%s",
                    candidate_idx + 1, budget_runs, cfg_id)
        for seed_idx in range(seeds_per_config):
            seed = base_seed + (candidate_idx + 1) * 100 + seed_idx
            run_id = _make_run_id("cur", candidate_idx, seed_idx)
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            result, score = _execute_one(
                trainer=trainer, config=cur_config, dataset_toml=dataset_toml,
                max_steps=max_steps_per_run, seed=seed, run_dir=run_dir,
                launcher=launcher, scorer=scorer, run_id=run_id,
                sample_prompts=effective_sample_prompts, sample_every_n_steps=sample_every_n_steps,
                judge_prompts=judge_prompts, n_in_dist_prompts=n_in_dist_prompts,
            )
            entry = _record(
                ledger, run_id=run_id, role="curated", cfg_id=cfg_id,
                config_dict=config_dict, seed=seed, seed_idx=seed_idx,
                result=result, score=score,
                trainer_name=trainer.name, search_space_name=space.name,
            )
            history.append(entry)
            candidate_history.append(entry)
            if score.disqualified:
                n_disqualified += 1
            logger.info("  curated seed=%d score=%s dq=%s duration=%.1fs",
                        seed_idx, score.score, score.disqualified, result.duration_s)
        candidate_idx += 1

    while True:
        if stop_event is not None and stop_event.is_set():
            logger.info("stop requested — exiting loop after %d candidates", candidate_idx)
            break
        if controller.should_stop(candidate_history, budget_runs=budget_runs):
            break
        knobs = controller.next_config(space, candidate_history)
        try:
            space.validate(knobs)
        except ValueError as e:
            logger.error("controller produced invalid sample: %s", e)
            break
        config = trainer.config_from_dict(knobs)
        config_dict = config.to_dict()
        cfg_id = config_id(config_dict)
        logger.info("running candidate %d/%d cfg_id=%s knobs=%s",
                    candidate_idx + 1, budget_runs, cfg_id, _short_knobs(knobs))
        for seed_idx in range(seeds_per_config):
            seed = base_seed + (candidate_idx + 1) * 100 + seed_idx
            run_id = _make_run_id("cand", candidate_idx, seed_idx)
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            result, score = _execute_one(
                trainer=trainer, config=config, dataset_toml=dataset_toml,
                max_steps=max_steps_per_run, seed=seed, run_dir=run_dir,
                launcher=launcher, scorer=scorer, run_id=run_id,
                sample_prompts=effective_sample_prompts, sample_every_n_steps=sample_every_n_steps,
                judge_prompts=judge_prompts, n_in_dist_prompts=n_in_dist_prompts,
            )
            entry = _record(
                ledger, run_id=run_id, role="candidate", cfg_id=cfg_id,
                config_dict=config_dict, seed=seed, seed_idx=seed_idx,
                result=result, score=score,
                trainer_name=trainer.name, search_space_name=space.name,
            )
            history.append(entry)
            candidate_history.append(entry)
            if score.disqualified:
                n_disqualified += 1
            logger.info("  seed=%d score=%s dq=%s duration=%.1fs",
                        seed_idx, score.score, score.disqualified, result.duration_s)
        candidate_idx += 1

    best = _pick_best_by_config(history)

    # Persist the session winner so subsequent sessions targeting the
    # same (trainer, search_space) pair can warm-start from it. We only
    # record when there's actually a winner — disqualified-only sessions
    # have nothing useful to remember.
    if best is not None and best.score is not None:
        # Multi-seed sessions report best by mean; capture the mean too.
        same_cfg = [
            h for h in history
            if h.score is not None and _config_id_from_history(h) == _config_id_from_history(best)
        ]
        mean_best = (
            sum(h.score for h in same_cfg) / len(same_cfg) if same_cfg else best.score
        )
        record_session_winner(
            trainer_name=trainer.name,
            search_space_name=space.name,
            best_config=dict(best.config),
            best_score=float(mean_best),
            n_seeds=len(same_cfg) or 1,
            db_path=history_db_path,
        )

    return OrchestrationResult(
        session_dir=output_dir,
        ledger_path=ledger.path,
        history=history,
        baseline_entry=baseline_entry,
        best_entry=best,
        n_completed=len(history),
        n_disqualified=n_disqualified,
    )


def _pick_best_by_config(history: list[LedgerEntry]) -> Optional[LedgerEntry]:
    """Best = config_id with lowest mean score across its seeds. Returns the
    first row of that config (so the caller has *something* to point at)."""
    by_cfg: dict[str, list[LedgerEntry]] = {}
    for h in history:
        if h.score is None:
            continue
        cfg_id = _config_id_from_history(h)
        by_cfg.setdefault(cfg_id, []).append(h)
    if not by_cfg:
        return None
    means = {cid: sum(h.score for h in rows) / len(rows) for cid, rows in by_cfg.items()}
    best_cid = min(means, key=means.get)
    return by_cfg[best_cid][0]


def _has_completed_setup(ledger: Ledger) -> bool:
    """True if the ledger has at least one role=setup row that exited 0."""
    for row in ledger.iter_rows():
        if row.get("role") == "setup" and row.get("exit_code") == 0:
            return True
    return False


def _config_id_from_history(h: LedgerEntry) -> str:
    return config_id(dict(h.config))


def _count_unique_configs(history: list[LedgerEntry], *, role_prefix: str) -> int:
    seen: set[str] = set()
    for h in history:
        if h.run_id.startswith(role_prefix):
            seen.add(_config_id_from_history(h))
    return len(seen)


def _short_knobs(knobs: dict) -> str:
    return str({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in knobs.items()})
