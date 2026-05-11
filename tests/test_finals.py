"""Tests for the finals stage — picking top-K configs from a stage-1 result
and that the wiring runs end-to-end with a mock trainer."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from bracket.orchestrator.finals import pick_top_k_configs, run_finals_stage
from bracket.orchestrator.loop import OrchestrationResult, orchestrate
from bracket.search.controller import LedgerEntry, RandomSearch
from bracket.search.space import FloatKnob, SearchSpace
from bracket.trainer.base import LaunchSpec, Trainer, TrainerConfig

from dataclasses import dataclass


@dataclass
class _MockConfig(TrainerConfig):
    target_loss: float = 0.3


class _MockTrainer(Trainer):
    name = "mock"

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(name="mock", knobs={"target_loss": FloatKnob(low=0.1, high=0.5)})

    def baseline_config(self) -> _MockConfig:
        return _MockConfig(target_loss=0.45)

    def config_from_dict(self, knobs: Mapping[str, Any]) -> _MockConfig:
        return _MockConfig(target_loss=float(knobs["target_loss"]))

    def prepare_run(
        self, *, run_dir, config, dataset_toml, max_steps, seed,
        sample_prompts: Optional[Path] = None,
        sample_every_n_steps: Optional[int] = None,
        save_every_n_steps: Optional[int] = None,
        save_state: bool = False,
        resume_from: Optional[Path] = None,
    ) -> LaunchSpec:
        run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
        target = config.target_loss   # type: ignore[attr-defined]
        helper = run_dir / "fake.py"
        helper.write_text(
            "from pathlib import Path\n"
            "from tensorboard.summary.writer.event_file_writer import EventFileWriter\n"
            "from tensorboard.compat.proto.event_pb2 import Event\n"
            "from tensorboard.compat.proto.summary_pb2 import Summary\n"
            "import time\n"
            f"target = {target}\n"
            f"d = Path(r'{run_dir}/logs/run_x'); d.mkdir(parents=True, exist_ok=True)\n"
            "w = EventFileWriter(str(d), max_queue_size=512, flush_secs=0)\n"
            "for s in range(1, 50):\n"
            "    loss = 0.5 + (target - 0.5) * (s / 50)\n"
            "    w.add_event(Event(wall_time=time.time(), step=s, summary=Summary(value=[Summary.Value(tag='loss/current', simple_value=loss)])))\n"
            "w.close()\n",
            encoding="utf-8",
        )
        import sys
        return LaunchSpec(
            cmd=[sys.executable, str(helper)],
            cwd=run_dir, env=dict(os.environ),
            output_dir=run_dir / "out", logging_dir=run_dir / "logs",
            tfevents_glob=str(run_dir / "logs" / "**" / "events.out.tfevents.*"),
        )


def test_pick_top_k_returns_lowest_score_configs():
    history = [
        LedgerEntry(run_id="cand-000-s0", config={"x": 1}, score=0.3, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),
        LedgerEntry(run_id="cand-001-s0", config={"x": 2}, score=0.1, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),
        LedgerEntry(run_id="cand-002-s0", config={"x": 3}, score=0.2, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),
        LedgerEntry(run_id="baseline-000-s0", config={"x": 0}, score=0.05, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),  # baseline excluded
        LedgerEntry(run_id="cand-003-s0", config={"x": 4}, score=None, error="dq",
                    duration_s=1.0, n_steps=10, timestamp=0.0),  # dq excluded
    ]
    stage1 = OrchestrationResult(
        session_dir=Path("/tmp"), ledger_path=Path("/tmp/l.jsonl"),
        history=history, baseline_entry=None, best_entry=None,
        n_completed=5, n_disqualified=1,
    )
    top2 = pick_top_k_configs(stage1, k=2)
    assert len(top2) == 2
    assert top2[0]["x"] == 2  # lowest score
    assert top2[1]["x"] == 3


def test_pick_top_k_handles_empty_or_all_disqualified():
    stage1 = OrchestrationResult(
        session_dir=Path("/tmp"), ledger_path=Path("/tmp/l.jsonl"),
        history=[], baseline_entry=None, best_entry=None,
        n_completed=0, n_disqualified=0,
    )
    assert pick_top_k_configs(stage1, k=3) == []


def test_pick_top_k_aggregates_multi_seed_by_mean():
    history = [
        LedgerEntry(run_id="cand-000-s0", config={"x": 1}, score=0.10, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),
        LedgerEntry(run_id="cand-000-s1", config={"x": 1}, score=0.30, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),  # mean = 0.20
        LedgerEntry(run_id="cand-001-s0", config={"x": 2}, score=0.15, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),
        LedgerEntry(run_id="cand-001-s1", config={"x": 2}, score=0.17, error=None,
                    duration_s=1.0, n_steps=10, timestamp=0.0),  # mean = 0.16
    ]
    stage1 = OrchestrationResult(
        session_dir=Path("/tmp"), ledger_path=Path("/tmp/l.jsonl"),
        history=history, baseline_entry=None, best_entry=None,
        n_completed=4, n_disqualified=0,
    )
    top1 = pick_top_k_configs(stage1, k=1)
    assert len(top1) == 1
    assert top1[0]["x"] == 2  # lower mean wins, even though x=1 had a lower single value


def test_run_finals_stage_end_to_end(tmp_path: Path):
    trainer = _MockTrainer()
    out = tmp_path / "session"
    stage1 = orchestrate(
        trainer=trainer, dataset_toml=tmp_path / "ds.toml",
        output_dir=out, controller=RandomSearch(seed=1),
        budget_runs=3, max_steps_per_run=40, max_wall_seconds_per_run=60,
    )
    assert stage1.n_completed >= 4

    finals = run_finals_stage(
        trainer=trainer, stage1=stage1,
        dataset_toml=tmp_path / "ds.toml", output_dir=out,
        top_k=2, finals_max_steps=40, finals_max_wall_seconds=60,
        finals_seeds_per_config=1,
    )
    assert len(finals.promoted_config_ids) == 2
    assert finals.best_finalist is not None
    # The ledger should now contain finalist rows alongside the stage-1 ones
    rows = [
        __import__("json").loads(l) for l in
        (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    finalists = [r for r in rows if r.get("role") == "finalist"]
    assert len(finalists) == 2
