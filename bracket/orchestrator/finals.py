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

logger = logging.getLogger(__name__)


@dataclass
class FinalsResult:
    promoted_config_ids: list[str]
    finals_history: list[LedgerEntry]
    best_finalist: Optional[LedgerEntry]


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
            result, score = _execute_one(
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
    return FinalsResult(
        promoted_config_ids=promoted_cids,
        finals_history=finals_history,
        best_finalist=best_finalist,
    )
