"""Finals stage — promote top-K stage-1 configs to longer runs.

Why: stage 1 ranks configs cheaply (e.g. 300 steps each on a small subset).
Cheap rankings can disagree with rankings at production scale. The finals
stage rescores the top-K from stage 1 at a higher step budget (e.g. 1500-3000)
and re-ranks. The final "winner" is the lowest-scoring config in stage 2.

This is essentially a 2-bracket Hyperband — full ASHA is more elaborate
(many fidelity rungs, async promotion) but for our budgets it's overkill.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from bracket.judge.base import PairwiseJudge, SampleJudge, parse_judge_prompts_file
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

logger = logging.getLogger(__name__)


@dataclass
class FinalsResult:
    promoted_config_ids: list[str]
    finals_history: list[LedgerEntry]
    best_finalist: Optional[LedgerEntry]
    # Markdown fragment from the Bradley-Terry tournament when
    # ``enable_pairwise_finals`` is True. Empty string otherwise.
    pairwise_leaderboard_md: str = ""


def pick_top_k_configs(stage1: OrchestrationResult, k: int) -> list[dict]:
    """Return the top-K candidate configs (lowest mean score across seeds).

    Excludes baseline; excludes disqualified configs. If fewer than K configs
    were scored, returns however many there are.
    """
    by_cfg: dict[str, list[LedgerEntry]] = {}
    for h in stage1.history:
        if h.run_id.startswith("baseline-"):
            continue
        if h.score is None:
            continue
        cid = config_id(dict(h.config))
        by_cfg.setdefault(cid, []).append(h)
    means = {cid: sum(h.score for h in rows) / len(rows) for cid, rows in by_cfg.items()}
    ranked = sorted(means.keys(), key=means.get)
    out: list[dict] = []
    seen_ids: set[str] = set()
    for cid in ranked[:k]:
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        out.append(dict(by_cfg[cid][0].config))
    return out


def run_finals_stage(
    *,
    trainer: Trainer,
    stage1: OrchestrationResult,
    dataset_toml: Path,
    output_dir: Path,
    top_k: int,
    finals_max_steps: int,
    finals_max_wall_seconds: int,
    finals_seeds_per_config: int = 2,
    base_seed: int = 42,
    sample_prompts: Optional[Path] = None,
    sample_every_n_steps: Optional[int] = None,
    sample_judge: Optional[SampleJudge] = None,
    loss_weight: float = 0.3,
    sample_weight: float = 0.7,
    mirror_stdout: bool = False,
    pairwise_judge: Optional[PairwiseJudge] = None,
    on_launcher_ready: Optional[Callable[[RunLauncher], None]] = None,
) -> FinalsResult:
    """Run the top-K stage-1 candidates at a higher step budget; append rows
    to the same ledger so the proof report covers both stages."""
    output_dir = Path(output_dir).resolve()
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(output_dir / "ledger.jsonl")
    launcher = RunLauncher(
        max_wall_seconds=finals_max_wall_seconds, mirror_stdout=mirror_stdout,
    )
    if on_launcher_ready is not None:
        try:
            on_launcher_ready(launcher)
        except Exception:  # noqa: BLE001
            logger.exception("on_launcher_ready callback raised")
    scorer = Scorer(
        sample_judge=sample_judge if sample_prompts is not None else None,
        loss_weight=loss_weight, sample_weight=sample_weight,
    )
    judge_prompts = parse_judge_prompts_file(sample_prompts) if sample_prompts else None

    promoted = pick_top_k_configs(stage1, k=top_k)
    if not promoted:
        logger.info("finals: no eligible stage-1 configs to promote; skipping")
        return FinalsResult(promoted_config_ids=[], finals_history=[], best_finalist=None)

    logger.info("finals: promoting top %d stage-1 configs to %d-step runs",
                len(promoted), finals_max_steps)
    finals_history: list[LedgerEntry] = []
    promoted_cids: list[str] = []
    for finalist_idx, cfg_dict in enumerate(promoted):
        config = trainer.config_from_dict(cfg_dict)
        cid = config_id(cfg_dict)
        promoted_cids.append(cid)
        for seed_idx in range(finals_seeds_per_config):
            seed = base_seed + 10000 + finalist_idx * 100 + seed_idx
            run_id = f"final-{finalist_idx:03d}-s{seed_idx}-{int(time.time())}"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info("finals: cand=%d seed_idx=%d cfg_id=%s run_id=%s",
                        finalist_idx, seed_idx, cid, run_id)
            result, score, _live = _execute_one(
                trainer=trainer, config=config, dataset_toml=dataset_toml,
                max_steps=finals_max_steps, seed=seed, run_dir=run_dir,
                launcher=launcher, scorer=scorer, run_id=run_id,
                sample_prompts=sample_prompts, sample_every_n_steps=sample_every_n_steps,
                judge_prompts=judge_prompts,
            )
            entry = _record(
                ledger, run_id=run_id, role="finalist", cfg_id=cid,
                config_dict=cfg_dict, seed=seed, seed_idx=seed_idx,
                result=result, score=score,
            )
            finals_history.append(entry)
            logger.info("  finalist score=%s dq=%s duration=%.1fs",
                        score.score, score.disqualified, result.duration_s)

    scored = [h for h in finals_history if h.score is not None]
    best_finalist = min(scored, key=lambda h: h.score) if scored else None

    pairwise_md = ""
    if pairwise_judge is not None and judge_prompts and len(promoted) >= 2:
        pairwise_md = _run_pairwise_tournament(
            finals_history=finals_history,
            promoted_cids=promoted_cids,
            prompts=judge_prompts,
            judge=pairwise_judge,
            runs_dir=runs_dir,
        )
        if pairwise_md:
            (output_dir / "pairwise_leaderboard.md").write_text(
                pairwise_md, encoding="utf-8",
            )

    return FinalsResult(
        promoted_config_ids=promoted_cids,
        finals_history=finals_history,
        best_finalist=best_finalist,
        pairwise_leaderboard_md=pairwise_md,
    )


def _run_pairwise_tournament(
    *,
    finals_history: list[LedgerEntry],
    promoted_cids: list[str],
    prompts: list[str],
    judge: PairwiseJudge,
    runs_dir: Path,
) -> str:
    """Execute the Bradley-Terry round-robin over finalist configs.

    Picks the best-seeded run per config as the representative when a
    finalist has multiple seeds — the orchestrator typically keeps the
    sample images in that run's sample_dir. Returns the rendered markdown
    fragment (empty when too few finalists have judgeable samples).
    """
    from bracket.proof.pairwise_ranking import (
        bootstrap_elo_cis,
        render_leaderboard_md,
        run_tournament,
    )

    representative_run: dict[str, LedgerEntry] = {}
    for h in finals_history:
        if h.score is None:
            continue
        cid = config_id(dict(h.config))
        cur = representative_run.get(cid)
        if cur is None or (cur.score is not None and h.score < cur.score):
            representative_run[cid] = h

    # Collect the sample image paths per competitor. The finals scorer
    # writes them under <run_dir>/output/sample (sd-scripts) or similar
    # for musubi. We probe both conventions and pick whichever the
    # trainer actually populated.
    images_for: dict[str, list[Path]] = {}
    for cid in promoted_cids:
        h = representative_run.get(cid)
        if h is None:
            continue
        sample_dir = _locate_sample_dir(runs_dir, h.run_id)
        if sample_dir is None or not sample_dir.exists():
            continue
        images_for[cid] = _pick_one_image_per_prompt(sample_dir, prompts)

    eligible = [cid for cid in promoted_cids if cid in images_for and len(images_for[cid]) == len(prompts)]
    if len(eligible) < 2:
        logger.warning(
            "pairwise finals: not enough eligible finalists with samples; "
            "skipping tournament (have %d, need >=2).", len(eligible),
        )
        return ""

    logger.info(
        "pairwise finals: %d finalists × %d prompts = %d judge calls",
        len(eligible), len(prompts), len(eligible) * (len(eligible) - 1) // 2 * len(prompts),
    )
    matches = run_tournament(eligible, prompts, images_for, judge)
    if not matches:
        return ""
    entries = bootstrap_elo_cis(matches, n_resamples=1000, seed=0)
    return render_leaderboard_md(entries)


def _locate_sample_dir(runs_dir: Path, run_id: str) -> Optional[Path]:
    """Derive the sample_dir for a finalist by convention.

    sd-scripts writes to ``<run>/output/sample``; musubi-tuner writes to
    ``<run>/output/samples``. We probe both and return whichever exists.
    """
    base = runs_dir / run_id / "output"
    for sub in ("sample", "samples"):
        candidate = base / sub
        if candidate.exists():
            return candidate
    return None


def _pick_one_image_per_prompt(sample_dir: Path, prompts: list[str]) -> list[Path]:
    """Return one image path per prompt index, or fewer when some are missing."""
    from bracket.orchestrator.scorer import _pair_samples_with_prompts

    pair = _pair_samples_with_prompts(sample_dir, prompts)
    out: list[Optional[Path]] = [None] * len(prompts)
    for img, prompt in pair.items():
        try:
            idx = prompts.index(prompt)
        except ValueError:
            continue
        if out[idx] is None:
            out[idx] = img
    return [p for p in out if p is not None]
