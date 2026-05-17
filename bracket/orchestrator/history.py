"""Persistent history of past Bracket session winners.

A session's "winner" is the non-disqualified config with the lowest mean
score across its seeds (same definition the report and the search
controller use). We persist one row per completed session into a small
SQLite database under ``~/.cache/bracket/history.db`` so the next
session targeting the same trainer + search-space pair can warm-start
from configurations that have already proven themselves.

The database is intentionally tiny — text-only columns, no JSON column
type — so it survives Python upgrades and is trivially diffable. The
config dict round-trips as JSON in ``best_config_json``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("bracket.orchestrator.history")


DEFAULT_HISTORY_DB = Path.home() / ".cache" / "bracket" / "history.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_winners (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    trainer_name TEXT NOT NULL,
    search_space_name TEXT NOT NULL,
    best_config_json TEXT NOT NULL,
    best_score REAL NOT NULL,
    n_seeds INTEGER NOT NULL DEFAULT 1,
    dataset_image_count INTEGER NOT NULL DEFAULT 0,
    dataset_caption_count INTEGER NOT NULL DEFAULT 0,
    dataset_size_bytes INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_session_winners_lookup
    ON session_winners(trainer_name, search_space_name, best_score);
"""


def _resolve_db_path(db_path: Optional[Path]) -> Path:
    if db_path is not None:
        return Path(db_path)
    override = os.environ.get("BRACKET_HISTORY_DB")
    if override:
        return Path(override).expanduser()
    return DEFAULT_HISTORY_DB


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_session_winner(
    *,
    trainer_name: str,
    search_space_name: str,
    best_config: dict,
    best_score: float,
    n_seeds: int = 1,
    dataset_image_count: int = 0,
    dataset_caption_count: int = 0,
    dataset_size_bytes: int = 0,
    db_path: Optional[Path] = None,
    ts: Optional[float] = None,
) -> None:
    """Append one row to the history DB. Idempotent if the exact same
    (trainer, search_space, config) is recorded multiple times — we keep
    every entry; ranking happens at lookup time.

    Failures are logged and swallowed: history bookkeeping must never
    crash a session that otherwise completed."""
    resolved = _resolve_db_path(db_path)
    try:
        with _connect(resolved) as conn:
            conn.execute(
                "INSERT INTO session_winners ("
                "ts, trainer_name, search_space_name, best_config_json, "
                "best_score, n_seeds, dataset_image_count, "
                "dataset_caption_count, dataset_size_bytes"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    float(ts if ts is not None else time.time()),
                    str(trainer_name),
                    str(search_space_name),
                    json.dumps(best_config, sort_keys=True, default=str),
                    float(best_score),
                    int(n_seeds),
                    int(dataset_image_count),
                    int(dataset_caption_count),
                    int(dataset_size_bytes),
                ),
            )
    except (sqlite3.Error, OSError) as e:
        logger.warning(
            "history: failed to record session winner for %s/%s: %s",
            trainer_name, search_space_name, e,
        )


def lookup_warmstart_configs(
    trainer_name: str,
    search_space_name: str,
    *,
    n: int = 3,
    db_path: Optional[Path] = None,
) -> list[dict]:
    """Return the top-``n`` best configs from history for this
    ``(trainer, search_space)`` pair, lowest-score-first.

    Each returned dict is the raw config dict that was stored at session
    completion. Duplicates (sessions that happened to land on the same
    config twice) are collapsed to one entry — only the first occurrence
    by score is kept.

    On any error (corrupt DB, permission issue, missing file) returns an
    empty list; warm-starting from history is a nice-to-have, not load
    bearing.
    """
    resolved = _resolve_db_path(db_path)
    if not resolved.exists():
        return []
    try:
        with _connect(resolved) as conn:
            cur = conn.execute(
                "SELECT best_config_json, best_score FROM session_winners "
                "WHERE trainer_name = ? AND search_space_name = ? "
                "ORDER BY best_score ASC LIMIT ?",
                (str(trainer_name), str(search_space_name), int(max(n, 0) * 4 + 4)),
            )
            rows = cur.fetchall()
    except (sqlite3.Error, OSError) as e:
        logger.warning(
            "history: failed to read warmstart configs for %s/%s: %s",
            trainer_name, search_space_name, e,
        )
        return []

    seen: set[str] = set()
    out: list[dict] = []
    for cfg_json, _score in rows:
        if cfg_json in seen:
            continue
        seen.add(cfg_json)
        try:
            out.append(json.loads(cfg_json))
        except json.JSONDecodeError:
            continue
        if len(out) >= n:
            break
    return out
