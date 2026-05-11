"""Background orchestration session for the UI.

Runs orchestrate() in a worker thread, exposes a thread-safe SessionState
that the UI polls. The session captures progress from logging by attaching
a custom handler — no changes to the orchestration core needed.

Threading discipline:
  - Use RLock so the same thread re-entering (e.g. via log emit while we
    hold the lock) doesn't deadlock.
  - The log handler must NEVER block indefinitely waiting for the lock —
    if contended, drop the line. Keeping the logger fast is more important
    than every line landing in the tail.
  - snapshot() returns a freshly-constructed SessionState copy so callers
    can read it lock-free.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from bracket.orchestrator.loop import OrchestrationResult


@dataclass
class SessionState:
    status: str = "idle"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_dir: Optional[Path] = None
    error_message: Optional[str] = None
    log_tail: deque = field(default_factory=lambda: deque(maxlen=400))
    last_run_summary: str = ""
    completed_runs: int = 0
    total_runs_target: int = 0
    result: Optional[OrchestrationResult] = None
    finals_started: bool = False
    # True when start() was called with a judge wired up (LMStudio +
    # sample_prompts). Persists for the session's lifetime so a Stop
    # before the first scored run still surfaces the right message.
    judge_configured: bool = False
    # The original (full) dataset_toml path passed to /session/start.
    # Different from `output_dir/subset/dataset.toml` which is the
    # auto-subset used for short search runs. Needed by the "promote"
    # flow so a full training run uses the full dataset.
    source_dataset_toml: Optional[Path] = None

    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


class _StateLogHandler(logging.Handler):
    """Routes orchestrator log lines into SessionState.log_tail.

    Uses try-acquire with a short timeout — if we can't get the lock quickly
    we drop the line rather than block the calling thread's logging path
    (which could cause cascading deadlocks with pytest's log capture).
    """

    def __init__(self, state: SessionState, lock: threading.RLock) -> None:
        super().__init__()
        self.state = state
        self.lock = lock

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            return
        # Best-effort. deque.append is atomic in CPython; the only contention
        # would be with snapshot() which copies the deque under the same lock.
        if not self.lock.acquire(timeout=0.25):
            return
        try:
            self.state.log_tail.append(line)
            msg = record.getMessage()
            if "score=" in msg and ("baseline" in msg or "candidate" in msg or "finalist" in msg):
                self.state.completed_runs += 1
                self.state.last_run_summary = msg
        finally:
            self.lock.release()


class OrchestrationSession:
    """One-shot wrapper around orchestrate() (+ optional finals) running in a thread."""

    def __init__(self) -> None:
        self.state = SessionState()
        # RLock so the same thread re-entering (logging emit while we hold)
        # doesn't deadlock.
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._handler: Optional[_StateLogHandler] = None
        # Stop signal — UI sets this; orchestrate() and the launcher poll it.
        self.stop_event = threading.Event()
        # Direct reference to the active launcher so a Stop click can kill the
        # currently-running subprocess instantly without waiting for the
        # poll-tick.
        self._active_launcher = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> bool:
        """Signal the running session to stop after the current candidate.

        Also kills the currently-running trainer subprocess immediately, so
        you don't have to wait the rest of the wall-time budget. Returns True
        if a session was running; False if there was nothing to stop.
        """
        if not self.is_running():
            return False
        self.stop_event.set()
        with self._lock:
            self.state.status = "stopping"
            self.state.log_tail.append("[user] stop requested")
        launcher = self._active_launcher
        if launcher is not None:
            try:
                launcher.kill_current()
            except Exception:
                pass
        return True

    def start(
        self,
        run_fn: Callable[[], OrchestrationResult],
        *,
        finals_fn: Optional[Callable[[OrchestrationResult], None]] = None,
        output_dir: Path,
        total_runs_target: int,
        judge_configured: bool = False,
        source_dataset_toml: Optional[Path] = None,
    ) -> None:
        if self.is_running():
            raise RuntimeError("session already running")
        # Reset stop signal for a new run.
        self.stop_event = threading.Event()
        with self._lock:
            self.state = SessionState(
                status="running", started_at=time.time(),
                output_dir=Path(output_dir), total_runs_target=total_runs_target,
                judge_configured=judge_configured,
                source_dataset_toml=Path(source_dataset_toml) if source_dataset_toml else None,
            )
            handler = _StateLogHandler(self.state, self._lock)
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self._handler = handler
        for name in ("bracket", "orchestrate"):
            logging.getLogger(name).addHandler(handler)
            logging.getLogger(name).setLevel(logging.INFO)

        def _runner() -> None:
            try:
                result = run_fn()
                with self._lock:
                    self.state.result = result
                if finals_fn is not None:
                    with self._lock:
                        self.state.finals_started = True
                    finals_fn(result)
                with self._lock:
                    self.state.status = "done"
                    self.state.finished_at = time.time()
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self.state.status = "error"
                    self.state.error_message = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    self.state.finished_at = time.time()
            finally:
                for name in ("bracket", "orchestrate"):
                    try:
                        logging.getLogger(name).removeHandler(handler)
                    except Exception:
                        pass

        self._thread = threading.Thread(target=_runner, name="bracket-session", daemon=True)
        self._thread.start()

    def snapshot(self) -> SessionState:
        # Copy the mutable bits under the lock; everything else is plain values.
        with self._lock:
            log_copy: deque = deque(self.state.log_tail, maxlen=self.state.log_tail.maxlen)
            return SessionState(
                status=self.state.status,
                started_at=self.state.started_at,
                finished_at=self.state.finished_at,
                output_dir=self.state.output_dir,
                error_message=self.state.error_message,
                log_tail=log_copy,
                last_run_summary=self.state.last_run_summary,
                completed_runs=self.state.completed_runs,
                total_runs_target=self.state.total_runs_target,
                result=self.state.result,
                finals_started=self.state.finals_started,
                judge_configured=self.state.judge_configured,
                source_dataset_toml=self.state.source_dataset_toml,
            )
