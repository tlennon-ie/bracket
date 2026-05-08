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
