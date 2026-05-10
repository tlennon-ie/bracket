"""Tests for the self-update API surface and the underlying helper.

No real network calls — :func:`bracket.updater._fetch_latest_release` is
monkeypatched to simulate GitHub responses, and :func:`trigger_update` is
exercised via a stubbed ``subprocess.Popen`` so we never actually fork the
installer.
"""

from __future__ import annotations

import time
from typing import Iterator
from urllib import error as urlerror

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from bracket import updater
from bracket.api import server as api_server
from bracket.api.server import create_app, get_session
from bracket.ui.session import OrchestrationSession


@pytest.fixture
def fresh_session() -> Iterator[OrchestrationSession]:
    yield OrchestrationSession()


@pytest.fixture
def app(fresh_session: OrchestrationSession) -> FastAPI:
    app = create_app(cors_origins=["http://localhost:5173"])
    app.dependency_overrides[get_session] = lambda: fresh_session
    api_server._SESSION = fresh_session  # type: ignore[attr-defined]
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_update_cache() -> Iterator[None]:
    """Force a fresh GitHub fetch in every test."""

    updater._cache_lock_value = None
    updater._cache_fetched_at = 0.0
    yield
    updater._cache_lock_value = None
    updater._cache_fetched_at = 0.0


# ───────────────────────── semver helpers ─────────────────────────


@pytest.mark.parametrize(
    "latest, current, expected",
    [
        ("0.2.0", "0.1.0", True),
        ("v0.2.0", "0.1.0", True),
        ("0.1.0", "0.1.0", False),
        ("0.1.0", "0.2.0", False),
        ("0.1.0-rc1", "0.1.0", False),  # pre-release suffix ignored, equal core
        ("1.0.0", "0.99.99", True),
    ],
)
def test_is_newer(latest: str, current: str, expected: bool) -> None:
    assert updater._is_newer(latest, current) is expected


# ───────────────────────── check_for_update ─────────────────────────


def test_check_for_update_reports_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updater, "__version__", "0.1.0")
    monkeypatch.setattr(
        updater,
        "_fetch_latest_release",
        lambda slug, timeout=5.0: {
            "tag_name": "v0.2.0",
            "html_url": "https://github.com/tlennon-ie/bracket/releases/tag/v0.2.0",
            "body": "## Notes\n- New trainer support.",
        },
    )

    result = updater.check_for_update(force=True)
    assert result.update_available is True
    assert result.latest_version == "0.2.0"
    assert result.current_version == "0.1.0"
    assert result.release_url and result.release_url.endswith("v0.2.0")
    assert result.release_notes and "Notes" in result.release_notes
    assert result.error is None


def test_check_for_update_no_update_when_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updater, "__version__", "0.2.0")
    monkeypatch.setattr(
        updater,
        "_fetch_latest_release",
        lambda slug, timeout=5.0: {"tag_name": "v0.2.0", "html_url": "u", "body": ""},
    )
    result = updater.check_for_update(force=True)
    assert result.update_available is False
    assert result.latest_version == "0.2.0"


def test_check_for_update_swallows_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> dict:
        raise urlerror.URLError("offline")

    monkeypatch.setattr(updater, "_fetch_latest_release", boom)
    monkeypatch.setattr(updater, "_fetch_default_branch_tip", boom)
    result = updater.check_for_update(force=True)
    assert result.update_available is False
    assert result.error is not None and "URLError" in result.error
    assert result.latest_version is None


def test_check_for_update_falls_back_to_commits_when_no_releases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If /releases/latest 404s (repo has no published releases),
    fall back to comparing local HEAD against the branch tip."""

    def releases_404(*_a: object, **_k: object) -> dict:
        raise urlerror.HTTPError(
            "https://api.github.com/.../releases/latest",
            404, "Not Found", hdrs=None, fp=None,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(updater, "_fetch_latest_release", releases_404)
    monkeypatch.setattr(
        updater,
        "_fetch_default_branch_tip",
        lambda slug, branch, timeout=5.0: {
            "sha": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "html_url": "https://github.com/x/y/commit/deadbeef",
            "commit": {"message": "feat: shiny new thing"},
        },
    )
    monkeypatch.setattr(updater, "_local_head_sha", lambda: "0123456789abcdef" * 2 + "01234")

    result = updater.check_for_update(force=True)
    assert result.update_available is True
    assert result.latest_version == "main@deadbee"
    assert result.current_version.endswith("(0123456)")
    assert result.release_url and "deadbeef" in result.release_url
    assert result.release_notes and "shiny" in result.release_notes


def test_check_via_commits_no_update_when_sha_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = "ab" * 20
    monkeypatch.setattr(
        updater,
        "_fetch_default_branch_tip",
        lambda slug, branch, timeout=5.0: {
            "sha": sha, "html_url": "u", "commit": {"message": ""},
        },
    )
    monkeypatch.setattr(updater, "_local_head_sha", lambda: sha)
    result = updater._check_via_commits("x/y", "main", time.time())
    assert result.update_available is False


def test_check_via_commits_no_update_when_local_sha_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If git isn't available locally, we can't compare — don't pester user."""

    monkeypatch.setattr(
        updater,
        "_fetch_default_branch_tip",
        lambda slug, branch, timeout=5.0: {
            "sha": "ab" * 20, "html_url": "u", "commit": {"message": ""},
        },
    )
    monkeypatch.setattr(updater, "_local_head_sha", lambda: None)
    result = updater._check_via_commits("x/y", "main", time.time())
    assert result.update_available is False


def test_check_for_update_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_fetch(slug: str, timeout: float = 5.0) -> dict:
        calls.append(1)
        return {"tag_name": "0.1.0", "html_url": "u", "body": ""}

    monkeypatch.setattr(updater, "__version__", "0.1.0")
    monkeypatch.setattr(updater, "_fetch_latest_release", fake_fetch)

    updater.check_for_update(force=False)
    updater.check_for_update(force=False)
    assert len(calls) == 1, "second call within TTL should be served from cache"

    updater.check_for_update(force=True)
    assert len(calls) == 2, "force=True must bypass the cache"


# ───────────────────────── /api/update/check ─────────────────────────


def test_api_update_check_returns_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(updater, "__version__", "0.1.0")
    monkeypatch.setattr(
        updater,
        "_fetch_latest_release",
        lambda slug, timeout=5.0: {
            "tag_name": "v0.5.0",
            "html_url": "https://example.com/r/v0.5.0",
            "body": "release body",
        },
    )
    res = client.get("/api/update/check?force=true")
    assert res.status_code == 200
    body = res.json()
    assert body["update_available"] is True
    assert body["latest_version"] == "0.5.0"
    assert body["current_version"] == "0.1.0"
    assert body["release_url"].endswith("v0.5.0")


def test_api_update_check_handles_offline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        updater,
        "_fetch_latest_release",
        lambda *a, **k: (_ for _ in ()).throw(urlerror.URLError("offline")),
    )
    res = client.get("/api/update/check?force=true")
    assert res.status_code == 200, "offline must NOT fail the endpoint"
    body = res.json()
    assert body["update_available"] is False
    assert body["error"] is not None


# ───────────────────────── /api/update/apply ─────────────────────────


def test_api_update_apply_spawns_script(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, cmd, **kwargs) -> None:  # noqa: ANN001
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    fake_script = tmp_path / "update.sh"
    fake_script.write_text("#!/bin/sh\necho hi\n")

    monkeypatch.setattr(updater.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(updater, "_update_script", lambda: fake_script)
    monkeypatch.setattr(updater, "_repo_root", lambda: tmp_path)

    res = client.post("/api/update/apply")
    assert res.status_code == 200
    body = res.json()
    assert body["spawned"] is True
    assert body["script"] == str(fake_script)
    assert body["log_path"] and body["log_path"].endswith("update.log")
    assert "cmd" in captured


def test_api_update_apply_returns_404_when_script_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(
        updater, "_update_script", lambda: tmp_path / "does-not-exist.sh"
    )
    res = client.post("/api/update/apply")
    assert res.status_code == 404
    body = res.json()
    assert "Update script not found" in body["detail"]
