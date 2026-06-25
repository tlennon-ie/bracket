"""Parse an ai-toolkit SQLite ``loss_log.db`` into LossFrames.

[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) writes NO TensorBoard
event files. When launched with ``logging.use_ui_logger: true`` its ``UILogger``
(``toolkit/logging_aitk.py``) instead records training metrics into a structured
SQLite database at ``<training_folder>/<name>/loss_log.db`` (WAL mode). The DB is
closed when the run finishes, so Bracket reads it **post-hoc and read-only**.

Schema (authoritative, from ``UILogger``)::

    steps(step INTEGER PRIMARY KEY, wall_time REAL NOT NULL)
    metrics(step INTEGER, key TEXT, value_real REAL, value_text TEXT,
            PRIMARY KEY(step, key))
    metric_keys(key TEXT PRIMARY KEY, first_seen_step INTEGER, last_seen_step INTEGER)

Loss is logged under ``key = 'loss'`` (``SDTrainer`` emits
``loss_dict = {'loss': ...}``). Learning rate, if present, is under
``key = 'lr'`` or ``'learning_rate'`` — we query for both and use whichever
exists (``lr`` is optional/display-only in ``LossFrame``).

This is the SQLite analog of ``bracket.log_loss_reader`` (the stdout-log reader
for LTX-2): it produces the same ``LossFrame`` schema (EMA-smoothed,
tfevents-only fields left ``None``) so the scorer reuses its existing
``_score_frames`` path unchanged. Like that reader it is **post-hoc only** —
there is no live tail for ai-toolkit runs yet (live pruning stays
tfevents-based).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from bracket.ema import EMASmoother
from bracket.frame import LossFrame

logger = logging.getLogger("bracket.sqlite_loss_reader")

# The metric keys under which ai-toolkit may log the learning rate, in
# preference order. ``lr`` is the common case; ``learning_rate`` is tolerated
# in case a logging config spells it out.
_LR_KEYS: tuple[str, ...] = ("lr", "learning_rate")

# Ascending-by-step join of the loss metric with its wall_time. ``LEFT JOIN``
# so a loss row whose step is somehow absent from ``steps`` still yields a frame
# (wall_time NULL → filled with 0.0 below). The lr lookup is a correlated
# subquery over the candidate lr keys, returning the first matching value.
#
# We deliberately do NOT filter ``m.value_real IS NOT NULL`` on the loss row:
# SQLite stores a Python ``float('nan')`` as NULL (NaN is not representable as a
# stored REAL), so ai-toolkit's ``UILogger`` — which writes the raw loss float
# straight into ``value_real`` — lands a NaN loss as a ``'loss'`` row with NULL
# ``value_real``. Dropping those would silently discard the gradient-explosion
# signal and score the run off its last finite frame. Instead we keep every
# ``'loss'`` row and map a NULL ``value_real`` back to ``float('nan')`` so the
# scorer's ``nan_loss`` disqualification fires. The lr subquery still filters
# NULLs (a missing lr is genuinely "no value", not a sentinel).
_LOSS_QUERY = (
    "SELECT m.step AS step, m.value_real AS loss, s.wall_time AS wall_time, ("
    "  SELECT lm.value_real FROM metrics lm "
    "  WHERE lm.step = m.step AND lm.key IN ({lr_placeholders}) "
    "  AND lm.value_real IS NOT NULL LIMIT 1"
    ") AS lr "
    "FROM metrics m LEFT JOIN steps s ON s.step = m.step "
    "WHERE m.key = 'loss' "
    "ORDER BY m.step ASC"
)


class SqliteLossReader:
    """Parse a completed ai-toolkit ``loss_log.db`` into a list of LossFrames.

    Opens the database **read-only** (``mode=ro`` URI) so a half-written WAL or
    a file Bracket has no business mutating is never touched. Returns ``[]`` for
    a missing / unopenable DB or one with no ``'loss'`` rows — it never raises,
    mirroring ``LogLossReader.read_all()``.
    """

    def __init__(self, db_path: Path | str, *, ema_alpha: float = 0.05) -> None:
        self._db_path = Path(db_path)
        self._ema_alpha = ema_alpha

    def read_all(self) -> list[LossFrame]:
        if not self._db_path.exists():
            logger.debug("loss db does not exist: %s", self._db_path)
            return []

        rows = self._fetch_rows()
        if not rows:
            return []

        # Build frames in ascending step order (the query already sorts), feeding
        # the EMA in step order so the smoothed curve matches the other readers.
        smoother = EMASmoother(alpha=self._ema_alpha)
        frames: list[LossFrame] = []
        for step, raw_loss, wall_time, lr in rows:
            # A NULL loss value_real is SQLite's storage of a NaN loss (see the
            # _LOSS_QUERY comment) — map it back so the scorer DQs the run.
            loss_value = float("nan") if raw_loss is None else float(raw_loss)
            smoothed = smoother.update(loss_value)
            frames.append(
                LossFrame(
                    step=int(step),
                    raw_loss=loss_value,
                    smoothed_loss=float(smoothed),
                    lr=(float(lr) if lr is not None else None),
                    wall_time=(float(wall_time) if wall_time is not None else 0.0),
                    timestep=None,
                    timestep_bucket=None,
                    grad_norm=None,
                )
            )
        return frames

    def _fetch_rows(self) -> list[tuple[int, Optional[float], Optional[float], Optional[float]]]:
        """Return ``(step, loss, wall_time, lr)`` rows ascending by step, or [].

        ``loss`` may be ``None``: ai-toolkit stores ``float('nan')`` as SQL NULL,
        and the caller maps that back to ``float('nan')`` so the scorer's
        ``nan_loss`` disqualification fires.

        Opens the DB read-only and tolerates WAL. Any sqlite/OS error (missing
        tables, corrupt/non-sqlite file, locked DB) yields ``[]`` rather than
        propagating — a broken telemetry file must not abort scoring.
        """
        # ``mode=ro`` requires the URI form; ``as_posix`` keeps Windows paths
        # valid in the ``file:`` URI.
        uri = f"file:{self._db_path.as_posix()}?mode=ro"
        placeholders = ",".join("?" for _ in _LR_KEYS)
        query = _LOSS_QUERY.format(lr_placeholders=placeholders)
        try:
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.Error:
            logger.exception("failed to open loss db %s (read-only)", self._db_path)
            return []
        try:
            cursor = conn.execute(query, _LR_KEYS)
            return [
                (row[0], row[1], row[2], row[3]) for row in cursor.fetchall()
            ]
        except sqlite3.Error:
            logger.exception("failed to read loss rows from %s", self._db_path)
            return []
        finally:
            conn.close()


def parse_sqlite_loss(
    db_path: Path | str, *, ema_alpha: float = 0.05,
) -> list[LossFrame]:
    """Module-level convenience wrapper around ``SqliteLossReader.read_all()``."""
    return SqliteLossReader(db_path, ema_alpha=ema_alpha).read_all()
