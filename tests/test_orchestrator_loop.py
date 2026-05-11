"""End-to-end test of the orchestrator loop using a mock Trainer that fakes
both the subprocess and the tfevents file. Verifies that the loop runs the
baseline + N candidates, records each to the ledger, and picks the best."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary

from bracket.orchestrator.loop import orchestrate
from bracket.search.controller import RandomSearch
from bracket.search.space import FloatKnob, SearchSpace
from bracket.trainer.base import LaunchSpec, Trainer, TrainerConfig


def _emit_tfevents(dir_: Path, final_loss: float, n_steps: int = 80) -> None:
    """Synthesise a tfevents file inside dir_/logs/ that converges to final_loss."""
    log_subdir = dir_ / "logs" / "run_x"
    log_subdir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(log_subdir), max_queue_size=512, flush_secs=0)
    base = time.time()
    initial = 0.5
    for step in range(1, n_steps + 1):
        loss = initial + (final_loss - initial) * (step / n_steps)
        writer.add_event(Event(
            wall_time=base + step * 0.01, step=step,
            summary=Summary(value=[Summary.Value(tag="loss/current", simple_value=loss)]),
        ))
    writer.close()


@dataclass
class MockConfig(TrainerConfig):
    target_loss: float = 0.4


class MockTrainer(Trainer):
    name = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(name="mock", knobs={"target_loss": FloatKnob(low=0.1, high=0.5)})

    def baseline_config(self) -> MockConfig:
        return MockConfig(target_loss=0.45)

    def config_from_dict(self, knobs: Mapping[str, Any]) -> MockConfig:
        return MockConfig(target_loss=float(knobs["target_loss"]))

    def prepare_run(
        self, *, run_dir: Path, config: TrainerConfig, dataset_toml: Path,
        max_steps: int, seed: int, sample_prompts: Optional[Path] = None,
        sample_every_n_steps: Optional[int] = None,
        save_every_n_steps: Optional[int] = None,
        save_state: bool = False,
        resume_from: Optional[Path] = None,
    ) -> LaunchSpec:
        self.calls += 1
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = config  # type: ignore[assignment]
        target = cfg.target_loss  # type: ignore[attr-defined]
        # The "subprocess" is a tiny python -c that synthesises a tfevents file.
        import sys
        # Cannot capture target via closure into subprocess, so embed value as literal.
        # Use a temp Python script written to disk to avoid quoting nightmares.
        helper = run_dir / "fake_train.py"
        helper.write_text(
            "from pathlib import Path\n"
            "import time\n"
            "from tensorboard.summary.writer.event_file_writer import EventFileWriter\n"
            "from tensorboard.compat.proto.event_pb2 import Event\n"
            "from tensorboard.compat.proto.summary_pb2 import Summary\n"
            f"target = {target}\n"
            "n_steps = 80\n"
            f"d = Path(r'{run_dir}/logs/run_x')\n"
            "d.mkdir(parents=True, exist_ok=True)\n"
            "w = EventFileWriter(str(d), max_queue_size=512, flush_secs=0)\n"
            "import time as _t\n"
            "base = _t.time()\n"
            "initial = 0.5\n"
            "for step in range(1, n_steps+1):\n"
            "    loss = initial + (target - initial) * (step / n_steps)\n"
            "    w.add_event(Event(wall_time=base+step*0.01, step=step,\n"
            "        summary=Summary(value=[Summary.Value(tag='loss/current', simple_value=loss)])))\n"
            "w.close()\n",
            encoding="utf-8",
        )
        return LaunchSpec(
            cmd=[sys.executable, str(helper)],
            cwd=run_dir,
            env=dict(os.environ),
            output_dir=run_dir / "out",
            logging_dir=run_dir / "logs",
            tfevents_glob=str(run_dir / "logs" / "**" / "events.out.tfevents.*"),
        )


def test_orchestrator_runs_baseline_then_candidates_and_picks_best(tmp_path: Path):
    trainer = MockTrainer()
    controller = RandomSearch(seed=1)
    out = tmp_path / "session"
    result = orchestrate(
        trainer=trainer,
        dataset_toml=tmp_path / "ds.toml",
        output_dir=out,
        controller=controller,
        budget_runs=3,
        max_steps_per_run=80,
        max_wall_seconds_per_run=60,
    )
    assert result.n_completed == 4  # baseline + 3 candidates
    assert result.baseline_entry is not None
    assert result.best_entry is not None
    # Best should not be disqualified
    assert result.best_entry.score is not None
    # Ledger written and parseable
    rows = list(__import__('json').loads(line)
                for line in result.ledger_path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert len(rows) == 4
    assert rows[0]["role"] == "baseline"


class _MockTrainerWithCurated(MockTrainer):
    """MockTrainer + a fixed list of curated configs for warm-start tests."""

    def curated_configs(self):
        return [MockConfig(target_loss=0.30), MockConfig(target_loss=0.32)]


def test_orchestrator_runs_curated_before_controller_picks(tmp_path: Path):
    """Curated warm-start configs must execute first, in declared order, and
    only after they're done should the controller take over the remaining
    budget."""
    trainer = _MockTrainerWithCurated()
    out = tmp_path / "session"
    orchestrate(
        trainer=trainer,
        dataset_toml=tmp_path / "ds.toml",
        output_dir=out,
        controller=RandomSearch(seed=1),
        budget_runs=4,            # 2 curated + 2 controller
        max_steps_per_run=80,
        max_wall_seconds_per_run=60,
    )
    rows = [__import__('json').loads(line)
            for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    roles = [r["role"] for r in rows if r["role"] != "setup"]
    assert roles[0] == "baseline"
    assert roles[1] == "curated"
    assert roles[2] == "curated"
    assert roles[3] == "candidate"
    assert roles[4] == "candidate"
    # Curated configs must appear in declared order (target_loss=0.30, then 0.32)
    curated_rows = [r for r in rows if r["role"] == "curated"]
    assert curated_rows[0]["config"]["target_loss"] == 0.30
    assert curated_rows[1]["config"]["target_loss"] == 0.32


def test_orchestrator_n_curated_zero_skips_warmstart(tmp_path: Path):
    """n_curated=0 preserves the legacy "controller-only" behaviour."""
    trainer = _MockTrainerWithCurated()
    out = tmp_path / "session"
    orchestrate(
        trainer=trainer, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=1), budget_runs=2,
        max_steps_per_run=80, max_wall_seconds_per_run=60,
        n_curated=0,
    )
    rows = [__import__('json').loads(line)
            for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    roles = [r["role"] for r in rows if r["role"] != "setup"]
    assert "curated" not in roles
    # baseline + 2 candidates
    assert roles == ["baseline", "candidate", "candidate"]


def test_orchestrator_n_curated_caps_warmstart(tmp_path: Path):
    """n_curated=1 runs only the first curated config, controller fills the rest."""
    trainer = _MockTrainerWithCurated()
    out = tmp_path / "session"
    orchestrate(
        trainer=trainer, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=1), budget_runs=3,
        max_steps_per_run=80, max_wall_seconds_per_run=60,
        n_curated=1,
    )
    rows = [__import__('json').loads(line)
            for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    roles = [r["role"] for r in rows if r["role"] != "setup"]
    curated_rows = [r for r in rows if r["role"] == "curated"]
    assert len(curated_rows) == 1
    assert curated_rows[0]["config"]["target_loss"] == 0.30  # first curated only
    assert roles.count("candidate") == 2


def test_orchestrator_resumable(tmp_path: Path):
    """Re-running orchestrate with the same output_dir should not re-run baseline.
    budget_runs is a *total* target — resumption continues toward it."""
    trainer = MockTrainer()
    out = tmp_path / "session"
    orchestrate(
        trainer=trainer, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=1), budget_runs=1,
        max_steps_per_run=40, max_wall_seconds_per_run=60,
    )
    # First session = baseline + 1 candidate = 2 calls
    assert trainer.calls == 2
    orchestrate(
        trainer=trainer, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=2), budget_runs=3,
        max_steps_per_run=40, max_wall_seconds_per_run=60,
    )
    # Second session: baseline skipped (already in ledger); already had 1 candidate;
    # budget_runs=3 means run 2 more candidates. Total calls: 2 + 2 = 4.
    assert trainer.calls == 4
