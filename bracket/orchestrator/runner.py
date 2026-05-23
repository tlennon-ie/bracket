"""Managed-subprocess launcher for one candidate training run.

Responsibilities:
  - Start the subprocess with the LaunchSpec's cmd / cwd / env.
  - Stream stdout+stderr to a per-run log file (and optionally to console).
  - Enforce a wall-time budget (kills cleanly if exceeded).
  - Locate the tfevents file the trainer wrote (post-hoc glob).
  - Return a RunResult that the scorer can consume.

Hardened against the things that have historically broken subprocess wrappers
on Windows: shell quoting (we never use shell=True), CTRL_BREAK on the child
process group (so accelerate's child threads die with the parent), and reading
stdout in a thread to avoid pipe-buffer deadlock.
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from bracket.trainer.base import LaunchSpec

logger = logging.getLogger(__name__)


# A live-check callback is polled every ``poll_interval`` seconds while the
# trainer is running. Returning a non-empty string asks the launcher to kill
# the subprocess and surface the reason via ``RunResult.error`` /
# ``killed_by_pruner``. Returning None / empty string means "keep running".
LiveCheck = Callable[[], Optional[str]]


@dataclass
class RunResult:
    run_id: str
    spec: LaunchSpec
    exit_code: Optional[int]      # None if killed before exit
    duration_s: float
    tfevents_path: Optional[Path] # None if no tfevents was produced
    log_path: Path
    error: Optional[str] = None
    killed_by_timeout: bool = False
    killed_by_pruner: bool = False
    pruner_reason: Optional[str] = None


class RunLauncher:
    def __init__(
        self,
        *,
        max_wall_seconds: int = 600,
        mirror_stdout: bool = False,
        stop_event: Optional["__import__('threading').Event"] = None,
    ) -> None:
        self.max_wall_seconds = max_wall_seconds
        self.mirror_stdout = mirror_stdout
        # Optional Event the UI can set to stop training; the launcher polls
        # it during proc.wait() and kills the subprocess if set.
        self.stop_event = stop_event
        self._current_proc: Optional[subprocess.Popen] = None

    def launch(
        self,
        run_id: str,
        spec: LaunchSpec,
        log_path: Path,
        *,
        live_check: Optional[LiveCheck] = None,
    ) -> RunResult:
        """Spawn the trainer with stdout redirected DIRECTLY to log_path.

        We deliberately do NOT use subprocess.PIPE for stdout. Reason: tqdm
        writes `\\r`-overwriting progress bars without newlines, which fills
        an OS pipe (~64KB on Windows) without emitting a `\\n`. Anything
        reading line-by-line then blocks, the pipe stays full, and the
        trainer's next stdout write blocks too — training freezes at
        thousands of seconds per step.

        Writing straight to a file: no pipe, no buffer to overflow, no
        deadlock. We give up live `mirror_stdout` (which we never used in
        practice anyway), and instead the Monitor tab reads the log file and
        the tfevents to surface progress in the UI.
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        creationflags = 0
        preexec_fn = None
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid

        # Open log file as binary write — trainer's bytes flow straight in.
        log_file = log_path.open("ab", buffering=0)
        try:
            log_file.write(f"# bracket run {run_id}\n".encode("utf-8"))
            log_file.write(f"# cwd: {spec.cwd}\n".encode("utf-8"))
            log_file.write(f"# cmd: {spec.cmd}\n#\n".encode("utf-8"))
            log_file.flush()
            try:
                proc = subprocess.Popen(
                    spec.cmd,
                    cwd=str(spec.cwd),
                    env=dict(spec.env),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    preexec_fn=preexec_fn,
                )
            except Exception as e:
                err = f"failed to spawn subprocess: {e!r}"
                log_file.write((err + "\n").encode("utf-8"))
                return RunResult(
                    run_id=run_id, spec=spec, exit_code=None,
                    duration_s=time.monotonic() - start,
                    tfevents_path=None, log_path=log_path, error=err,
                )

            self._current_proc = proc
            killed_by_timeout = False
            killed_by_user = False
            killed_by_pruner = False
            pruner_reason: Optional[str] = None
            deadline = time.monotonic() + self.max_wall_seconds
            poll_interval = 1.0
            try:
                while True:
                    try:
                        exit_code = proc.wait(timeout=poll_interval)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stop_event is not None and self.stop_event.is_set():
                            killed_by_user = True
                            logger.warning("run %s: stop requested by user; killing", run_id)
                            self._kill(proc)
                            try:
                                exit_code = proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                exit_code = None
                            break
                        if live_check is not None:
                            try:
                                reason = live_check()
                            except Exception:  # noqa: BLE001 — live_check must never abort the run
                                logger.exception("run %s: live_check raised", run_id)
                                reason = None
                            if reason:
                                killed_by_pruner = True
                                pruner_reason = str(reason)
                                logger.warning(
                                    "run %s: killed by pruner — %s", run_id, pruner_reason,
                                )
                                self._kill(proc)
                                try:
                                    exit_code = proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    exit_code = None
                                break
                        if time.monotonic() >= deadline:
                            killed_by_timeout = True
                            logger.warning("run %s exceeded %ds; killing", run_id, self.max_wall_seconds)
                            self._kill(proc)
                            try:
                                exit_code = proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                exit_code = None
                            break
            finally:
                self._current_proc = None
        finally:
            try:
                log_file.close()
            except Exception:
                pass

        duration = time.monotonic() - start
        tfevents = self._find_tfevents(spec.tfevents_glob)
        if killed_by_user:
            err = "stopped_by_user"
        elif killed_by_pruner:
            err = f"pruned:{pruner_reason}" if pruner_reason else "pruned"
        elif killed_by_timeout:
            err = "timeout"
        elif exit_code == 0:
            err = None
        else:
            err = f"exit_code={exit_code}"
        return RunResult(
            run_id=run_id,
            spec=spec,
            exit_code=exit_code,
            duration_s=duration,
            tfevents_path=tfevents,
            log_path=log_path,
            killed_by_timeout=killed_by_timeout,
            killed_by_pruner=killed_by_pruner,
            pruner_reason=pruner_reason,
            error=err,
        )

    def kill_current(self) -> bool:
        """External signal to kill whatever's running RIGHT NOW.

        Returns True if a process was killed, False if nothing was running.
        Safe to call from any thread.
        """
        proc = self._current_proc
        if proc is None or proc.poll() is not None:
            return False
        self._kill(proc)
        return True

    @staticmethod
    def _kill(proc: subprocess.Popen) -> None:
        try:
            if os.name == "nt":
                # On Windows, the trainer process spawns torch DataLoader
                # workers and `accelerate` worker processes that do NOT share
                # a Win32 job object with the parent — `proc.kill()` only
                # terminates the top-level Python, leaving GPU-owning
                # workers running for many more steps until they fail their
                # next IPC rendezvous with the dead parent. That's the
                # "stop takes ages and keeps training" symptom.
                #
                # Strategy: soft-signal the process group with
                # CTRL_BREAK_EVENT first (gives Python a chance to flush),
                # then hard-kill the entire process tree via taskkill /T.
                import signal
                try:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                except Exception:  # noqa: BLE001
                    # Process may have exited between poll() and the signal.
                    pass
                # Short grace period to let CTRL_BREAK propagate, then
                # force-kill the whole tree.
                time.sleep(1.0)
                if proc.poll() is None:
                    # /T = tree, /F = force. Don't raise on non-zero exit:
                    # taskkill returns 128 when the PID already exited.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                        check=False,
                    )
                    # Belt-and-braces: if taskkill couldn't find children
                    # (e.g. detached), still try the standard kill.
                    if proc.poll() is None:
                        try:
                            proc.kill()
                        except Exception:  # noqa: BLE001
                            pass
            else:
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                time.sleep(2.0)
                if proc.poll() is None:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception("error during subprocess kill")

    @staticmethod
    def _find_tfevents(pattern: str) -> Optional[Path]:
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            return None
        # If multiple, pick the largest (most events written) — a healthy run
        # produces one big file; a crashed one may leave a tiny header-only file.
        biggest = max(matches, key=lambda p: Path(p).stat().st_size if Path(p).exists() else 0)
        return Path(biggest)
