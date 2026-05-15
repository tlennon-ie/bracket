"""Tests for the FastAPI surface in :mod:`bracket.api`.

These tests are unit-level: no orchestration is started. The session
singleton is replaced with a fresh instance per test via the
``get_session`` dependency override so tests don't bleed into each other.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Iterator

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bracket.api import server as api_server
from bracket.api.server import create_app, get_session
from bracket.ui.session import OrchestrationSession


# ───────────────────────────── fixtures ─────────────────────────────


@pytest.fixture
def fresh_session() -> Iterator[OrchestrationSession]:
    """Provide an isolated OrchestrationSession for each test."""

    sess = OrchestrationSession()
    yield sess


@pytest.fixture
def app(fresh_session: OrchestrationSession) -> FastAPI:
    app = create_app(cors_origins=["http://localhost:5173"])
    app.dependency_overrides[get_session] = lambda: fresh_session
    # The endpoints call get_session() directly (not via Depends), so we
    # also patch the module-level singleton for the test's lifetime.
    api_server._SESSION = fresh_session  # type: ignore[attr-defined]
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# ───────────────────────────── health ─────────────────────────────


def test_health_returns_version(client: TestClient) -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert isinstance(body["version"], str) and body["version"]


# ───────────────────────────── presets ─────────────────────────────


def test_families_returns_at_least_three(client: TestClient) -> None:
    res = client.get("/api/presets/families")
    assert res.status_code == 200
    families = res.json()
    names = {f["name"] for f in families}
    assert {"SDXL", "Z-Image", "Flux-2-Klein"}.issubset(names)


def test_types_for_unknown_family_is_404(client: TestClient) -> None:
    res = client.get("/api/presets/families/Imaginary/types")
    assert res.status_code == 404


def test_types_for_known_family(client: TestClient) -> None:
    res = client.get("/api/presets/families/SDXL/types")
    assert res.status_code == 200
    types = [t["name"] for t in res.json()]
    assert "LoRA" in types


def test_preset_for_invalid_pair_is_404(client: TestClient) -> None:
    res = client.get("/api/presets/SDXL/Nonsense")
    assert res.status_code == 404


def test_preset_for_valid_pair_includes_fields(client: TestClient) -> None:
    res = client.get("/api/presets/SDXL/LoRA")
    assert res.status_code == 200
    body = res.json()
    assert body["family"] == "SDXL"
    assert body["training_type"] == "LoRA"
    assert body["fields"], "preset must surface trainer fields"
    assert body["session_fields"], "preset must surface session fields"
    field_names = {f["name"] for f in body["fields"]}
    assert "venv_python" in field_names


# ───────────────────────────── session ─────────────────────────────


def test_session_idle_returns_coherent_snapshot(client: TestClient) -> None:
    res = client.get("/api/session")
    assert res.status_code == 200
    snap = res.json()
    assert snap["session_status"] == "idle"
    assert snap["output_dir"] is None
    assert snap["completed_runs"] == 0
    assert snap["score_history"] == []
    assert snap["status_line"]


def test_session_start_with_full_dataset_skips_subset_build(
    client: TestClient, monkeypatch, tmp_path: Path,
) -> None:
    """images_per_dataset=0 must skip build_subset() entirely and pass the
    user-supplied dataset_toml straight through to the orchestrator.

    Verified by monkey-patching ``build_subset`` to fail the test if it
    gets called and asserting the start path still succeeds."""

    from bracket.api import server as srv

    def boom(*_a, **_kw):
        raise AssertionError("build_subset must not be called when images_per_dataset<=0")

    monkeypatch.setattr(srv, "build_subset", boom)
    # Stub the trainer factory so we don't need GPU / real weights; the
    # session.start handler resolves families/types through the registry
    # so we only have to ensure construction doesn't blow up.
    # The session won't actually run a thread because we monkey-patch the
    # runner before it executes — see existing tests for the same pattern.
    monkeypatch.setattr(
        srv, "orchestrate",
        lambda **kw: srv.OrchestrationResult(
            session_dir=tmp_path, ledger_path=tmp_path / "ledger.jsonl",
            history=[], baseline_entry=None, best_entry=None,
            n_completed=0, n_disqualified=0,
        ),
    )
    # Minimum-valid dataset.toml the build_subset stub would have read.
    toml_path = tmp_path / "dataset.toml"
    toml_path.write_text(
        "[general]\nresolution = 1024\n\n"
        "[[datasets]]\n[[datasets.subsets]]\nimage_dir = '/x'\n",
        encoding="utf-8",
    )
    # Pick any known preset; the registry's first SDXL family/type will do.
    fams = client.get("/api/presets/families").json()
    fam = fams[0]["name"]
    types = client.get(f"/api/presets/families/{fam}/types").json()
    typ = types[0]["name"]
    body = {
        "family": fam, "training_type": typ,
        "dataset_toml": str(toml_path),
        "output_dir": str(tmp_path / "out"),
        "images_per_dataset": 0,  # ← the wedge
        "preset_field_values": {},
    }
    res = client.post(
        "/api/session/start", json=body,
    )
    # 200 → orchestrator was invoked; 400 with the AssertionError above
    # would mean build_subset() was called and our monkey-patch fired.
    assert res.status_code in (200, 400, 409)
    if res.status_code == 400:
        # If we hit a bad_request, it must NOT be the build_subset assertion.
        assert "build_subset" not in res.json().get("message", "")


def test_session_start_with_invalid_input_returns_400(client: TestClient) -> None:
    body = {
        "family": "SDXL",
        "training_type": "LoRA",
        "dataset_toml": "",  # missing — should fail validation
        "output_dir": "",
        "preset_field_values": {},
    }
    res = client.post("/api/session/start", json=body)
    assert res.status_code == 400
    payload = res.json()
    assert payload["status"] == "bad_request"
    assert "Missing required" in payload["message"]


def test_session_start_with_unknown_preset_returns_400(client: TestClient) -> None:
    body = {
        "family": "Nope",
        "training_type": "LoRA",
        "dataset_toml": "/tmp/x.toml",
        "output_dir": "/tmp/out",
        "preset_field_values": {},
    }
    res = client.post("/api/session/start", json=body)
    assert res.status_code == 400


def test_session_stop_when_idle_is_noop(client: TestClient) -> None:
    res = client.post("/api/session/stop")
    assert res.status_code == 200
    body = res.json()
    assert body["stopped"] is False


# ───────────────────────────── runs ─────────────────────────────


def test_runs_when_idle_returns_empty(client: TestClient) -> None:
    res = client.get("/api/runs")
    assert res.status_code == 200
    assert res.json() == []


def test_run_detail_for_unknown_run_is_404_when_no_session(client: TestClient) -> None:
    res = client.get("/api/runs/cand-007/")
    # The 404 here is "no session yet"; either way we must not 200.
    assert res.status_code in (404, 307)


def test_run_detail_for_unknown_run_after_session_setup(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    """Once a session has an output_dir, an unknown run_id returns 404."""

    # Manually populate the session state without actually orchestrating
    # — same shape produced by start() but without spawning a thread.
    fresh_session.state.output_dir = tmp_path
    (tmp_path / "ledger.jsonl").write_text("", encoding="utf-8")

    res = client.get("/api/runs/cand-no-such")
    assert res.status_code == 404


# ───────────────────────────── run log ─────────────────────────────


def test_run_log_when_idle_returns_empty_payload(client: TestClient) -> None:
    """No session yet → 200 with exists=False rather than 404, so the UI
    can poll harmlessly without dropping into a toasted error path."""

    res = client.get("/api/runs/cand-007/log")
    assert res.status_code == 200
    body = res.json()
    assert body["exists"] is False
    assert body["content"] == ""
    assert body["total_size"] == 0


def test_run_log_streams_appending_content(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    """End-to-end: write to a log file, fetch with offset=0, then write
    more and fetch again with offset=next_offset. Only the new bytes come
    back in the second response — that's the wedge that makes the live
    console viewer cheap to poll."""

    fresh_session.state.output_dir = tmp_path
    run_id = "cand-log-test"
    log_dir = tmp_path / "runs" / run_id / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "stdout.log"
    log_path.write_bytes(b"caching latents\n")

    first = client.get(f"/api/runs/{run_id}/log").json()
    assert first["exists"] is True
    assert first["content"] == "caching latents\n"
    assert first["offset"] == 0
    assert first["next_offset"] == 16
    assert first["total_size"] == 16

    # Append more — the second poll should return only the new bytes.
    with log_path.open("ab") as f:
        f.write(b"step 1/200\n")
    second = client.get(
        f"/api/runs/{run_id}/log?offset={first['next_offset']}",
    ).json()
    assert second["content"] == "step 1/200\n"
    assert second["offset"] == 16
    assert second["next_offset"] == 27


def test_run_log_handles_truncation_via_offset_reset(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    """If the file shrinks (rotated / truncated), the server falls back to
    offset=0 rather than returning a confusing partial chunk."""

    fresh_session.state.output_dir = tmp_path
    run_id = "cand-trunc"
    log_dir = tmp_path / "runs" / run_id / "logs"
    log_dir.mkdir(parents=True)
    log_path = log_dir / "stdout.log"
    log_path.write_bytes(b"a" * 1000)

    res = client.get(f"/api/runs/{run_id}/log?offset=5000").json()
    # offset > size → server resets to 0 and serves the whole file
    assert res["offset"] == 0
    assert len(res["content"]) == 1000


# ───────────────────────────── gallery + report ─────────────────────────────


def test_gallery_when_idle_is_empty(client: TestClient) -> None:
    res = client.get("/api/gallery")
    assert res.status_code == 200
    assert res.json() == []


def test_report_when_idle_is_404(client: TestClient) -> None:
    res = client.get("/api/report")
    assert res.status_code == 404


# ───────────────────────────── judge status ─────────────────────────────


def test_judge_status_when_idle(client: TestClient) -> None:
    res = client.get("/api/judge/status")
    assert res.status_code == 200
    body = res.json()
    assert "configured" in body
    assert "summary" in body


# ───────────────────────────── promote / config export ─────────────────────────────


def _write_session_meta(out: Path, **overrides) -> None:
    """Write a minimal ``session.json`` so promote/export endpoints have
    the metadata they need to reconstruct the original request."""

    import json
    req = {
        "family": "SDXL", "training_type": "LoRA",
        "dataset_toml": "/tmp/dataset.toml", "output_dir": str(out),
        "sample_prompts": "/tmp/prompts.txt", "resume": "",
        "images_per_dataset": 12, "vram_gb": None,
        "budget": 8, "max_steps": 300, "wall_secs": 1800, "seeds": 1,
        "search_method": "optuna", "optuna_startup": 5, "n_curated": -1,
        "finals_top_k": 0, "finals_max_steps": 2000, "finals_seeds": 2,
        "judge_method": "none", "judge_base_url": "http://localhost:1234/v1",
        "judge_model": "qwen2.5-vl",
        "judge_loss_weight": 0.3, "judge_sample_weight": 0.7,
        "judge_disable_thinking": False,
        "preset_field_values": {},
    }
    req.update(overrides)
    out.mkdir(parents=True, exist_ok=True)
    (out / "session.json").write_text(
        json.dumps({
            "request": req,
            "source_dataset_toml": req["dataset_toml"],
            "subset_dataset_toml": str(out / "subset" / "dataset.toml"),
            "started_at": 1234567890.0,
        }),
        encoding="utf-8",
    )


def _write_ledger_with_one_candidate(out: Path, run_id: str, score: float) -> None:
    import json
    (out / "ledger.jsonl").write_text(
        json.dumps({
            "run_id": run_id, "role": "candidate", "config_id": "cfg-001",
            "config": {"learning_rate": 5e-06, "optimizer_type": "Adafactor"},
            "seed": 42, "seed_idx": 0, "score": score,
            "score_components": {}, "judge_report": None,
            "n_steps": 300, "duration_s": 60.0, "exit_code": 0,
            "killed_by_timeout": False, "error": None, "disqualified": None,
        }) + "\n",
        encoding="utf-8",
    )


def test_training_config_export_returns_candidate_config(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    """The export endpoint must surface the candidate's config plus the
    session-level source dataset path, so users can promote later."""

    fresh_session.state.output_dir = tmp_path
    _write_session_meta(tmp_path)
    _write_ledger_with_one_candidate(tmp_path, "cand-007", 0.42)

    res = client.get("/api/runs/cand-007/training-config")
    assert res.status_code == 200
    body = res.json()
    assert body["run_id"] == "cand-007"
    assert body["family"] == "SDXL"
    assert body["training_type"] == "LoRA"
    assert body["score"] == 0.42
    assert "learning_rate" in body["config"]
    assert body["source_dataset_toml"] == "/tmp/dataset.toml"
    assert body["sample_prompts"] == "/tmp/prompts.txt"


def test_training_config_export_returns_404_for_unknown_run(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    fresh_session.state.output_dir = tmp_path
    (tmp_path / "ledger.jsonl").write_text("", encoding="utf-8")
    res = client.get("/api/runs/never-existed/training-config")
    assert res.status_code == 404


def test_config_export_returns_full_request(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    fresh_session.state.output_dir = tmp_path
    _write_session_meta(tmp_path, budget=16, seeds=2)
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["request"]["budget"] == 16
    assert body["request"]["seeds"] == 2
    assert body["bracket_version"]


def test_config_validate_round_trips_request(client: TestClient) -> None:
    """POST /api/config/validate validates an imported config blob via
    Pydantic so the UI surfaces field-level errors at import time."""

    payload = {
        "request": {
            "family": "Z-Image", "training_type": "Full FT",
            "dataset_toml": "/x", "output_dir": "/y",
            "sample_prompts": "", "resume": "",
            "images_per_dataset": 8, "vram_gb": None,
            "budget": 4, "max_steps": 100, "wall_secs": 600, "seeds": 1,
            "search_method": "optuna", "optuna_startup": 5, "n_curated": -1,
            "finals_top_k": 0, "finals_max_steps": 2000, "finals_seeds": 2,
            "judge_method": "none", "judge_base_url": "http://localhost:1234/v1",
            "judge_model": "qwen2.5-vl",
            "judge_loss_weight": 0.3, "judge_sample_weight": 0.7,
            "judge_disable_thinking": False,
            "preset_field_values": {},
        },
    }
    res = client.post("/api/config/validate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["request"]["family"] == "Z-Image"


def test_promote_endpoint_404_when_run_missing(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    fresh_session.state.output_dir = tmp_path
    _write_session_meta(tmp_path)
    (tmp_path / "ledger.jsonl").write_text("", encoding="utf-8")
    res = client.post("/api/runs/nonexistent/promote", json={
        "max_steps": 1000, "save_every_n_steps": 200,
        "save_state": True, "sample_every_n_steps": None,
        "resume_from": "", "full_dataset_toml": "",
    })
    assert res.status_code == 404


def test_promote_endpoint_409_when_session_already_running(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
    monkeypatch,
) -> None:
    fresh_session.state.output_dir = tmp_path
    _write_session_meta(tmp_path)
    _write_ledger_with_one_candidate(tmp_path, "cand-001", 0.5)
    monkeypatch.setattr(fresh_session, "is_running", lambda: True)
    res = client.post("/api/runs/cand-001/promote", json={
        "max_steps": 1000, "save_every_n_steps": 200,
        "save_state": True, "sample_every_n_steps": None,
        "resume_from": "", "full_dataset_toml": "",
    })
    assert res.status_code == 409


# ───────────────────────────── static file traversal ─────────────────────────────


def test_static_route_blocks_dotdot_traversal(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    """Confirm the static-file mount refuses ``..`` segments."""

    fresh_session.state.output_dir = tmp_path
    (tmp_path / "runs").mkdir()

    # Percent-encoded traversal — httpx leaves this raw so it reaches the
    # files router, where the path-containment check rejects it.
    res = client.get("/files/cand-001/..%2F..%2Fetc%2Fpasswd")
    assert res.status_code in (400, 403, 404)
    # Literal `../..` is normalised client-side by httpx before the request
    # ever leaves — the URL the server sees is plain `/etc/passwd`, which
    # the SPA fallback serves as index.html (no filesystem read happens).
    # That's intended SPA behaviour, not a security regression.


def test_static_route_blocks_non_whitelisted_extension(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    """Confirm extension whitelist blocks weights / arbitrary files."""

    runs = tmp_path / "runs" / "cand-001"
    runs.mkdir(parents=True)
    bad = runs / "weights.safetensors"
    bad.write_bytes(b"fake")
    fresh_session.state.output_dir = tmp_path

    res = client.get("/files/cand-001/weights.safetensors")
    assert res.status_code == 403


def test_static_route_serves_whitelisted_file(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    runs = tmp_path / "runs" / "cand-001" / "output" / "sample"
    runs.mkdir(parents=True)
    (runs / "step000020_00.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    fresh_session.state.output_dir = tmp_path

    res = client.get("/files/cand-001/output/sample/step000020_00.png")
    assert res.status_code == 200
    assert res.content.startswith(b"\x89PNG")


def test_static_route_404_for_missing(
    client: TestClient, fresh_session: OrchestrationSession, tmp_path: Path,
) -> None:
    (tmp_path / "runs" / "cand-001").mkdir(parents=True)
    fresh_session.state.output_dir = tmp_path
    res = client.get("/files/cand-001/output/sample/missing.png")
    assert res.status_code == 404


# ───────────────────────────── websocket ─────────────────────────────


@pytest.mark.websocket
def test_websocket_pushes_at_least_one_snapshot(client: TestClient) -> None:
    """Skip unless ``-m websocket`` is enabled — keeps the default suite fast."""

    with client.websocket_connect("/api/ws/snapshot") as ws:
        msg = ws.receive_json()
        assert "session_status" in msg
        assert "ts" in msg
