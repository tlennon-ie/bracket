"""LMStudio pairwise judge — verdict-parsing tests + monkeypatched HTTP."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bracket.judge.lmstudio_pairwise import (
    LMStudioPairwiseConfig,
    LMStudioPairwiseJudge,
)


def test_parse_verdict_accepts_clean_json() -> None:
    out = LMStudioPairwiseJudge.parse_verdict(
        '{"winner": "a", "reason": "sharper"}'
    )
    assert out == "a"


def test_parse_verdict_strips_think_block() -> None:
    raw = (
        "<think>let me consider...</think>\n"
        '{"winner": "b", "reason": "better adherence"}'
    )
    assert LMStudioPairwiseJudge.parse_verdict(raw) == "b"


def test_parse_verdict_accepts_tie() -> None:
    assert LMStudioPairwiseJudge.parse_verdict('{"winner": "tie"}') == "tie"


def test_parse_verdict_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        LMStudioPairwiseJudge.parse_verdict("no JSON here")
    with pytest.raises(ValueError):
        LMStudioPairwiseJudge.parse_verdict('{"winner": "maybe"}')


def test_judge_pair_returns_tie_on_http_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network errors must not crash a tournament — pair falls back to tie."""

    def boom(*_a, **_kw):
        raise OSError("simulated network outage")

    monkeypatch.setattr(
        "urllib.request.urlopen", boom,
    )
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"\x89PNG")
    b.write_bytes(b"\x89PNG")
    judge = LMStudioPairwiseJudge()
    assert judge.judge_pair(a, b, "anything") == 0


class _FakeResp:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, *_a):  # noqa: D401
        return False

    def read(self) -> bytes:
        return self._body


def test_judge_pair_routes_winner_to_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stubbed LMStudio response with winner=a should return -1
    (a wins); winner=b returns +1; tie returns 0."""
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"\x89PNG")
    b.write_bytes(b"\x89PNG")

    def make_resp(winner: str) -> _FakeResp:
        body = {
            "choices": [{"message": {"content": json.dumps({"winner": winner})}}]
        }
        return _FakeResp(json.dumps(body).encode("utf-8"))

    judge = LMStudioPairwiseJudge(LMStudioPairwiseConfig(use_json_schema=False))

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_kw: make_resp("a"),
    )
    assert judge.judge_pair(a, b, "p") == -1

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_kw: make_resp("b"),
    )
    assert judge.judge_pair(a, b, "p") == 1

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_a, **_kw: make_resp("tie"),
    )
    assert judge.judge_pair(a, b, "p") == 0
