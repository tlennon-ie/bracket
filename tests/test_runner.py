"""Runner subprocess tests. Uses Python itself as the "trainer" so these run
on any platform without sd-scripts."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from bracket.orchestrator.runner import RunLauncher
from bracket.trainer.base import LaunchSpec


def _spec(cmd: list[str], cwd: Path, glob: str) -> LaunchSpec:
    return LaunchSpec(
        cmd=cmd, cwd=cwd, env=dict(os.environ),
        output_dir=cwd, logging_dir=cwd, tfevents_glob=glob,
    )


def test_runner_captures_exit_code_zero(tmp_path: Path):
    spec = _spec(
        [sys.executable, "-c", "print('hello')"],
        tmp_path, str(tmp_path / "events.out.tfevents.*"),
    )
    r = RunLauncher(max_wall_seconds=30).launch("r1", spec, tmp_path / "log.txt")
    assert r.exit_code == 0
    assert r.killed_by_timeout is False
    assert r.error is None
    assert (tmp_path / "log.txt").read_text(encoding="utf-8").find("hello") >= 0
    assert r.tfevents_path is None  # no events file produced


def test_runner_captures_nonzero_exit_code(tmp_path: Path):
    spec = _spec(
        [sys.executable, "-c", "import sys; sys.exit(7)"],
        tmp_path, str(tmp_path / "events.out.tfevents.*"),
    )
    r = RunLauncher(max_wall_seconds=30).launch("r1", spec, tmp_path / "log.txt")
    assert r.exit_code == 7
    assert r.error == "exit_code=7"


def test_runner_finds_tfevents_file_after_run(tmp_path: Path):
    # Have the "trainer" write a fake events.out.tfevents.* file.
    cmd = [
        sys.executable, "-c",
        f"open(r'{tmp_path}/events.out.tfevents.fake', 'wb').write(b'hello')",
    ]
    spec = _spec(cmd, tmp_path, str(tmp_path / "events.out.tfevents.*"))
    r = RunLauncher(max_wall_seconds=30).launch("r1", spec, tmp_path / "log.txt")
    assert r.exit_code == 0
    assert r.tfevents_path is not None
    assert r.tfevents_path.name.startswith("events.out.tfevents.")


@pytest.mark.skipif(os.name == "nt", reason="signal semantics differ on Windows; covered by integration")
def test_runner_kills_long_running_process(tmp_path: Path):
    spec = _spec(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        tmp_path, str(tmp_path / "events.out.tfevents.*"),
    )
    start = time.monotonic()
    r = RunLauncher(max_wall_seconds=2).launch("r1", spec, tmp_path / "log.txt")
    elapsed = time.monotonic() - start
    assert r.killed_by_timeout is True
    assert elapsed < 15  # killed quickly, not waited for the full 60s
