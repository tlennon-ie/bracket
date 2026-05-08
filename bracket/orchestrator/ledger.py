"""Append-only JSONL ledger of orchestrator runs.

One file per orchestration session. One row per candidate (incl. baseline).
Atomic line-by-line append + fsync so a crashed orchestrator doesn't corrupt
prior history; the loop can resume from the last good line.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from bracket.search.controller import LedgerEntry


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return str(value)
    return value


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, row: dict[str, Any]) -> None:
        line = json.dumps(_to_jsonable(row), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter([])
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Tolerate a partial trailing write from a crashed run.
                    continue
        return iter(rows)

    def to_history(self) -> list[LedgerEntry]:
        out: list[LedgerEntry] = []
        for row in self.iter_rows():
            score = row.get("score")
            if isinstance(score, str):
                score = None  # serialized inf/nan
            out.append(
                LedgerEntry(
                    run_id=row.get("run_id", ""),
                    config=row.get("config", {}),
                    score=score,
                    error=row.get("error"),
                    duration_s=float(row.get("duration_s", 0.0)),
                    n_steps=int(row.get("n_steps", 0)),
                    timestamp=float(row.get("timestamp", 0.0)),
                )
            )
        return out

    @staticmethod
    def now_ts() -> float:
        return time.time()
