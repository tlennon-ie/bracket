"""Tests for SampleJudge plumbing — base class aggregation, LMStudio JSON parsing,
prompt-file parsing. No real network calls; we use a stub judge for the
integration paths."""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.judge.base import (
    JudgeReport,
    SampleJudge,
    SampleJudgement,
    parse_judge_prompts_file,
)
from bracket.judge.lmstudio import LMStudioJudge, LMStudioJudgeConfig


def _write_image(p: Path) -> None:
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes-for-testing")


class _FixedJudge(SampleJudge):
    def __init__(self, score: float = 7.0) -> None:
        self.score = score

    def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
        return SampleJudgement(
            image_path=image_path, prompt=prompt,
            prompt_adherence=self.score, visual_quality=self.score,
            artifact_free=self.score, overall=self.score,
            raw_response="fake",
        )


class _FlakyJudge(SampleJudge):
    def __init__(self, fail_indices: set[int]) -> None:
        self.fail_indices = fail_indices
        self.calls = 0

    def judge_image(self, image_path: Path, prompt: str) -> SampleJudgement:
        i = self.calls
        self.calls += 1
        if i in self.fail_indices:
            raise RuntimeError("boom")
        return SampleJudgement(
            image_path=image_path, prompt=prompt,
            prompt_adherence=8.0, visual_quality=8.0, artifact_free=8.0,
            overall=8.0, raw_response="ok",
        )


def test_judge_run_aggregates_scores(tmp_path: Path):
    img1 = tmp_path / "a.png"; img2 = tmp_path / "b.png"
    _write_image(img1); _write_image(img2)
    judge = _FixedJudge(score=6.0)
    rep = judge.judge_run(tmp_path, {img1: "p1", img2: "p2"})
    assert rep.n_images == 2
    assert rep.n_failed == 0
    assert rep.mean_overall == pytest.approx(6.0)
    # Lower-is-better normalisation: 1 - 6/10 = 0.4
    assert rep.normalized_score == pytest.approx(0.4)


def test_judge_run_tolerates_per_image_failure(tmp_path: Path):
    img1 = tmp_path / "a.png"; img2 = tmp_path / "b.png"
    _write_image(img1); _write_image(img2)
    judge = _FlakyJudge(fail_indices={0})
    rep = judge.judge_run(tmp_path, {img1: "p1", img2: "p2"})
    assert rep.n_images == 2
    assert rep.n_failed == 1
    assert rep.mean_overall == pytest.approx(8.0)  # only the successful one is counted


def test_judge_run_skips_missing_images(tmp_path: Path):
    img1 = tmp_path / "a.png"
    _write_image(img1)
    missing = tmp_path / "missing.png"
    judge = _FixedJudge()
    rep = judge.judge_run(tmp_path, {img1: "p1", missing: "p2"})
    assert rep.n_images == 1


def test_judge_run_all_failed_returns_zero(tmp_path: Path):
    img1 = tmp_path / "a.png"
    _write_image(img1)
    judge = _FlakyJudge(fail_indices={0})
    rep = judge.judge_run(tmp_path, {img1: "p1"})
    assert rep.mean_overall == 0.0
    assert rep.n_failed == 1


def test_parse_judge_prompts_strips_flags_and_comments(tmp_path: Path):
    f = tmp_path / "p.txt"
    f.write_text(
        "# header\n"
        "\n"
        "a woman walking on a beach --w 1024 --h 1024 --s 20 --d 42 --l 4.0\n"
        "# another comment\n"
        "portrait of a man --w 1024 --h 1024\n",
        encoding="utf-8",
    )
    out = parse_judge_prompts_file(f)
    assert out == [
        "a woman walking on a beach",
        "portrait of a man",
    ]


def test_lmstudio_extract_choice_text_normal_response():
    body = (
        '{"choices":[{"message":{"content":'
        '"{\\"prompt_adherence\\": 8, \\"visual_quality\\": 7, \\"artifact_free\\": 9}"'
        "}}]}"
    )
    text = LMStudioJudge._extract_choice_text(body)
    assert "prompt_adherence" in text


def test_lmstudio_parse_scores_strips_thinking_block():
    raw = (
        "<think>let me consider...</think>\n"
        '{"prompt_adherence": 7, "visual_quality": 8, "artifact_free": 6}'
    )
    parsed = LMStudioJudge.parse_scores(raw)
    assert parsed == {"prompt_adherence": 7.0, "visual_quality": 8.0, "artifact_free": 6.0}


def test_lmstudio_parse_scores_handles_extra_text():
    raw = (
        "Here are the scores I think:\n"
        '{"prompt_adherence": 5, "visual_quality": 6, "artifact_free": 4}\n'
        "Hope that helps!"
    )
    parsed = LMStudioJudge.parse_scores(raw)
    assert parsed["prompt_adherence"] == 5.0


def test_lmstudio_parse_scores_rejects_garbage():
    with pytest.raises(ValueError):
        LMStudioJudge.parse_scores("the model said: i refuse")


def test_lmstudio_autoscale_brings_0_5_to_0_10():
    cfg = LMStudioJudgeConfig(autoscale_threshold=5.0)
    j = LMStudioJudge(cfg)
    out = j._normalise_to_10({"prompt_adherence": 3.0, "visual_quality": 5.0, "artifact_free": 4.0})
    # Top was 5.0 → factor 2; values double, clipped to 10
    assert out["visual_quality"] == pytest.approx(10.0)
    assert out["prompt_adherence"] == pytest.approx(6.0)


def test_lmstudio_no_autoscale_when_already_0_10():
    cfg = LMStudioJudgeConfig(autoscale_threshold=5.0)
    j = LMStudioJudge(cfg)
    out = j._normalise_to_10({"prompt_adherence": 9.0, "visual_quality": 7.0, "artifact_free": 8.0})
    assert out["prompt_adherence"] == 9.0


def test_lmstudio_judge_image_handles_network_error_gracefully(tmp_path: Path, monkeypatch):
    img = tmp_path / "a.png"; _write_image(img)
    j = LMStudioJudge(LMStudioJudgeConfig(base_url="http://127.0.0.1:1/v1", timeout_s=0.1))
    res = j.judge_image(img, "test prompt")
    assert res.error is not None
    assert res.overall == 0.0


def test_lmstudio_eject_invokes_lms_cli(monkeypatch, tmp_path):
    """eject() must shell out to `lms unload <identifier>`. The previous
    HTTP endpoint was removed by LMStudio and silently no-ops if used."""
    fake_lms = tmp_path / "lms.exe"
    fake_lms.write_text("#!/bin/sh\nexit 0\n")
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["timeout"] = kwargs.get("timeout")

        class _R:
            returncode = 0
            stdout = "Unloaded qwen3-vl-8b\n"
            stderr = ""

        return _R()

    import bracket.judge.lmstudio as lm
    monkeypatch.setattr(lm.LMStudioJudge, "_resolve_lms_binary",
                        staticmethod(lambda: str(fake_lms)))
    monkeypatch.setattr(lm.subprocess, "run", _fake_run)

    j = LMStudioJudge(LMStudioJudgeConfig(
        base_url="http://localhost:1234/v1", model="qwen3-vl-8b",
    ))
    j.eject()

    assert captured["cmd"] == [str(fake_lms), "unload", "qwen3-vl-8b"]
    assert captured["timeout"] == 30.0


def test_lmstudio_eject_warns_when_cli_missing(monkeypatch, caplog):
    """If `lms` isn't installed, eject() should log a hint, not raise."""
    import logging as _logging
    import bracket.judge.lmstudio as lm
    monkeypatch.setattr(lm.LMStudioJudge, "_resolve_lms_binary",
                        staticmethod(lambda: None))
    j = LMStudioJudge(LMStudioJudgeConfig(model="qwen3-vl-8b"))
    with caplog.at_level(_logging.WARNING, logger="bracket.judge.lmstudio"):
        j.eject()
    assert any("lms" in rec.getMessage() for rec in caplog.records)


def test_lmstudio_eject_swallows_subprocess_failures(monkeypatch, tmp_path):
    """A flaky LMStudio control plane must not abort the orchestration loop."""
    fake_lms = tmp_path / "lms"
    fake_lms.write_text("")

    def _boom(*_a, **_k):
        raise OSError("simulated cli crash")

    import bracket.judge.lmstudio as lm
    monkeypatch.setattr(lm.LMStudioJudge, "_resolve_lms_binary",
                        staticmethod(lambda: str(fake_lms)))
    monkeypatch.setattr(lm.subprocess, "run", _boom)
    j = LMStudioJudge(LMStudioJudgeConfig(model="qwen3-vl-8b"))
    j.eject()  # must not raise


def test_lmstudio_eject_handles_nonzero_exit(monkeypatch, tmp_path, caplog):
    """`lms unload` returns non-zero when the model isn't currently loaded
    — that's not fatal."""
    fake_lms = tmp_path / "lms"
    fake_lms.write_text("")

    class _R:
        returncode = 1
        stdout = ""
        stderr = "Model 'x' is not currently loaded"

    import bracket.judge.lmstudio as lm
    import logging as _logging
    monkeypatch.setattr(lm.LMStudioJudge, "_resolve_lms_binary",
                        staticmethod(lambda: str(fake_lms)))
    monkeypatch.setattr(lm.subprocess, "run", lambda *a, **k: _R())
    j = LMStudioJudge(LMStudioJudgeConfig(model="x"))
    with caplog.at_level(_logging.WARNING, logger="bracket.judge.lmstudio"):
        j.eject()
    assert any("returned 1" in r.getMessage() for r in caplog.records)


def test_lmstudio_resolve_lms_binary_respects_env(monkeypatch, tmp_path):
    """BRACKET_LMS_BIN env var must override PATH lookup."""
    binary = tmp_path / "custom-lms"
    binary.write_text("")
    monkeypatch.setenv("BRACKET_LMS_BIN", str(binary))
    assert LMStudioJudge._resolve_lms_binary() == str(binary)


def test_lmstudio_payload_includes_json_schema_response_format():
    """The default payload must request structured JSON output so the
    underlying llama.cpp grammar prevents the model from rambling."""
    j = LMStudioJudge(LMStudioJudgeConfig(model="any-vlm"))
    payload = j._build_payload("data:image/png;base64,xxx", "a cat")
    assert "response_format" in payload
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]["schema"]
    assert schema["required"] == ["prompt_adherence", "visual_quality", "artifact_free"]
    assert schema["properties"]["prompt_adherence"]["maximum"] == 10
    assert payload["max_tokens"] >= 2048, "thinking models need headroom for preamble + JSON"


def test_lmstudio_payload_omits_response_format_when_disabled():
    """Older LMStudio builds may reject response_format; opt-out exists."""
    j = LMStudioJudge(LMStudioJudgeConfig(model="any-vlm", use_json_schema=False))
    payload = j._build_payload("data:image/png;base64,xxx", "a cat")
    assert "response_format" not in payload


def test_lmstudio_payload_omits_chat_template_kwargs_by_default():
    """Default is to leave the chat template alone — only opt-in."""
    j = LMStudioJudge(LMStudioJudgeConfig(model="any-vlm"))
    payload = j._build_payload("data:image/png;base64,xxx", "a cat")
    assert "chat_template_kwargs" not in payload


def test_lmstudio_payload_sets_enable_thinking_false_when_opted_in():
    """The disable-thinking toggle must translate into the OpenAI-spec
    chat_template_kwargs field that LMStudio forwards to the Jinja
    chat template renderer."""
    j = LMStudioJudge(LMStudioJudgeConfig(model="any-vlm", enable_thinking=False))
    payload = j._build_payload("data:image/png;base64,xxx", "a cat")
    assert payload.get("chat_template_kwargs") == {"enable_thinking": False}


def test_lmstudio_payload_passes_enable_thinking_true_through():
    """Symmetric: explicit True must round-trip too, in case a user wants
    to force-enable thinking on a model whose template defaults to off."""
    j = LMStudioJudge(LMStudioJudgeConfig(model="any-vlm", enable_thinking=True))
    payload = j._build_payload("data:image/png;base64,xxx", "a cat")
    assert payload.get("chat_template_kwargs") == {"enable_thinking": True}
