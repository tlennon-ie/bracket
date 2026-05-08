import math
from pathlib import Path

from bracket.orchestrator.ledger import Ledger


def test_ledger_append_and_iter(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append({"run_id": "r1", "score": 0.42})
    ledger.append({"run_id": "r2", "score": 0.31})
    rows = list(ledger.iter_rows())
    assert [r["run_id"] for r in rows] == ["r1", "r2"]


def test_ledger_serializes_path_and_dataclass(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append({"path": tmp_path / "x", "nested": {"p": tmp_path}})
    rows = list(ledger.iter_rows())
    assert isinstance(rows[0]["path"], str)
    assert isinstance(rows[0]["nested"]["p"], str)


def test_ledger_handles_inf_and_nan(tmp_path: Path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append({"score": math.inf, "other": math.nan})
    rows = list(ledger.iter_rows())
    # inf/nan serialized as strings — JSON would otherwise reject them
    assert isinstance(rows[0]["score"], str)


def test_ledger_to_history_skips_partial_lines(tmp_path: Path):
    p = tmp_path / "l.jsonl"
    ledger = Ledger(p)
    ledger.append({"run_id": "r1", "score": 0.5, "config": {"a": 1}, "n_steps": 10,
                   "duration_s": 1.0, "timestamp": 0.0})
    # Append a malformed trailing line directly
    with p.open("a", encoding="utf-8") as f:
        f.write("{not valid json")
    history = ledger.to_history()
    assert len(history) == 1
    assert history[0].run_id == "r1"
    assert history[0].score == 0.5


def test_ledger_string_score_treated_as_unscored(tmp_path: Path):
    """When score serialised as 'inf' (string), to_history should mark it None."""
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append({"run_id": "r1", "score": math.inf, "n_steps": 0,
                   "duration_s": 0.0, "timestamp": 0.0, "config": {}})
    history = ledger.to_history()
    assert history[0].score is None
