"""Resume-time trainer/search-space mismatch guard.

If a session was started with trainer A and the user re-runs the same
output_dir with trainer B, candidate scores aren't comparable across
trainers. The orchestrator must refuse to mix them — better to fail loud
than silently produce a garbage report.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from bracket.orchestrator.loop import orchestrate
from bracket.search.controller import RandomSearch
from bracket.search.space import FloatKnob, SearchSpace
from bracket.trainer.base import LaunchSpec, Trainer, TrainerConfig


@dataclass
class _Cfg(TrainerConfig):
    target_loss: float = 0.4


class _MockTrainer(Trainer):
    def __init__(self, name: str = "mock-A") -> None:
        self.name = name
        self.calls = 0

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(name=f"{self.name}-space",
                           knobs={"target_loss": FloatKnob(low=0.1, high=0.5)})

    def baseline_config(self) -> _Cfg:
        return _Cfg(target_loss=0.45)

    def config_from_dict(self, knobs: Mapping[str, Any]) -> _Cfg:
        return _Cfg(target_loss=float(knobs["target_loss"]))

    def prepare_run(self, *, run_dir: Path, config: TrainerConfig, dataset_toml: Path,
                    max_steps: int, seed: int, sample_prompts: Optional[Path] = None,
                    sample_every_n_steps: Optional[int] = None) -> LaunchSpec:
        self.calls += 1
        run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
        target = getattr(config, "target_loss", 0.4)
        helper = run_dir / "fake_train.py"
        helper.write_text(
            "from pathlib import Path\nimport time\n"
            "from tensorboard.summary.writer.event_file_writer import EventFileWriter\n"
            "from tensorboard.compat.proto.event_pb2 import Event\n"
            "from tensorboard.compat.proto.summary_pb2 import Summary\n"
            f"target = {target}\n"
            f"d = Path(r'{run_dir}/logs/run_x'); d.mkdir(parents=True, exist_ok=True)\n"
            "w = EventFileWriter(str(d), max_queue_size=512, flush_secs=0)\n"
            "import time as _t\nbase = _t.time()\ninitial = 0.5\n"
            "for step in range(1, 51):\n"
            "    loss = initial + (target - initial) * (step / 50)\n"
            "    w.add_event(Event(wall_time=base+step*0.01, step=step,\n"
            "        summary=Summary(value=[Summary.Value(tag='loss/current', simple_value=loss)])))\n"
            "w.close()\n",
            encoding="utf-8",
        )
        import sys
        return LaunchSpec(
            cmd=[sys.executable, str(helper)], cwd=run_dir, env=dict(os.environ),
            output_dir=run_dir / "out", logging_dir=run_dir / "logs",
            tfevents_glob=str(run_dir / "logs" / "**" / "events.out.tfevents.*"),
        )


def test_resume_with_different_trainer_raises(tmp_path: Path):
    out = tmp_path / "session"
    a = _MockTrainer(name="trainer-A")
    orchestrate(
        trainer=a, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=1), budget_runs=1,
        max_steps_per_run=40, max_wall_seconds_per_run=30,
    )
    # Now resume with a different trainer
    b = _MockTrainer(name="trainer-B")
    with pytest.raises(RuntimeError, match="trainer mismatch"):
        orchestrate(
            trainer=b, dataset_toml=tmp_path / "ds.toml", output_dir=out,
            controller=RandomSearch(seed=2), budget_runs=2,
            max_steps_per_run=40, max_wall_seconds_per_run=30,
        )


def test_resume_with_same_trainer_works(tmp_path: Path):
    out = tmp_path / "session"
    a = _MockTrainer(name="trainer-A")
    orchestrate(
        trainer=a, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=1), budget_runs=1,
        max_steps_per_run=40, max_wall_seconds_per_run=30,
    )
    a2 = _MockTrainer(name="trainer-A")
    orchestrate(
        trainer=a2, dataset_toml=tmp_path / "ds.toml", output_dir=out,
        controller=RandomSearch(seed=2), budget_runs=2,
        max_steps_per_run=40, max_wall_seconds_per_run=30,
    )
    rows = [json.loads(line) for line in (out / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # All rows have the same trainer name
    assert all(r.get("trainer_name") == "trainer-A" for r in rows)
