"""Self-update helper.

Provides two pieces:

1. :func:`check_for_update` — first asks the GitHub Releases API for the
   latest tag and compares against ``bracket.__version__``. Repos without
   any published releases (the common case during early development) get a
   404 here; we then fall back to comparing the local ``HEAD`` SHA against
   the remote default-branch tip via the ``/commits`` endpoint. Result is
   the same shape either way, so the React UI doesn't care which path was
   used. Network failures are swallowed.

2. :func:`trigger_update` — spawns the platform-appropriate update script
   (``update.bat`` / ``update.sh``) detached from the parent process. The
   script does ``git pull``, rebuilds the Python venv + frontend, and
   re-launches the server. Returns immediately; the React frontend will see
   the WebSocket drop and reconnect once the new server is up.

The GitHub repo slug is read from the env var ``BRACKET_REPO_SLUG`` so
forks or self-hosted mirrors can override it; default points at the
upstream ``tlennon-ie/bracket`` repository. The default branch is read
from ``BRACKET_REPO_DEFAULT_BRANCH`` (defaults to ``main``).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from bracket import __version__

logger = logging.getLogger("bracket.updater")

DEFAULT_REPO_SLUG = "tlennon-ie/bracket"
DEFAULT_REPO_BRANCH = "main"
_CACHE_TTL_SECONDS = 60 * 30  # 30 minutes — GitHub anonymous API limit is 60/hr.

_cache_lock_value: Optional["UpdateCheck"] = None
_cache_fetched_at: float = 0.0


@dataclass(frozen=True)
class UpdateCheck:
    """Result of a single :func:`check_for_update` call."""

    current_version: str
    latest_version: Optional[str]
    update_available: bool
    release_url: Optional[str]
    release_notes: Optional[str]
    checked_at: float
    error: Optional[str] = None


def _repo_slug() -> str:
    return os.environ.get("BRACKET_REPO_SLUG", DEFAULT_REPO_SLUG).strip() or DEFAULT_REPO_SLUG


def _repo_branch() -> str:
    return os.environ.get("BRACKET_REPO_DEFAULT_BRANCH", DEFAULT_REPO_BRANCH).strip() or DEFAULT_REPO_BRANCH


def _local_head_sha() -> Optional[str]:
    """Resolve the current repo HEAD via ``git rev-parse``.

    Returns ``None`` if git isn't on PATH, the repo isn't a git checkout,
    or the call fails for any reason. We never want update detection to
    crash the API.
    """

    try:
        result = subprocess.run(  # noqa: S603 - "git" is a fixed argv[0]
            ["git", "rev-parse", "HEAD"],
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = (result.stdout or "").strip()
    return sha or None


def _normalize_tag(tag: str) -> str:
    """Strip a leading ``v`` so ``v0.1.0`` and ``0.1.0`` compare equal."""

    return tag.lstrip("vV").strip()


def _parse_semver(version: str) -> tuple[int, ...]:
    """Lenient semver parse — returns a tuple suitable for comparison.

    Non-numeric suffixes (``-rc1``, ``-dev``) are ignored so we don't
    misclassify a pre-release tag as "newer" than the stable.
    """

    core = _normalize_tag(version).split("-")[0].split("+")[0]
    parts: list[int] = []
    for chunk in core.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def _is_newer(latest: str, current: str) -> bool:
    return _parse_semver(latest) > _parse_semver(current)


def _gh_get(url: str, *, timeout: float = 5.0) -> dict:
    """GET a GitHub API URL and return the parsed JSON.

    Raises :class:`urlerror.HTTPError` (a subclass of URLError) on non-2xx —
    callers can catch ``HTTPError`` to detect 404 specifically and fall
    back to a different endpoint.
    """

    req = urlrequest.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"bracket/{__version__}",
        },
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - hardcoded host
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _fetch_latest_release(slug: str, *, timeout: float = 5.0) -> dict:
    """Hit ``/repos/{slug}/releases/latest``."""

    return _gh_get(f"https://api.github.com/repos/{slug}/releases/latest", timeout=timeout)


def _fetch_default_branch_tip(slug: str, branch: str, *, timeout: float = 5.0) -> dict:
    """Hit ``/repos/{slug}/commits/{branch}`` for the latest commit on the
    default branch. Used when the repo has no published Releases."""

    return _gh_get(f"https://api.github.com/repos/{slug}/commits/{branch}", timeout=timeout)


def _check_via_releases(slug: str, now: float) -> Optional[UpdateCheck]:
    """Try the Releases API; return ``None`` if the repo has no releases."""

    try:
        payload = _fetch_latest_release(slug)
    except urlerror.HTTPError as e:
        if e.code == 404:
            # No releases published yet — caller falls back to /commits.
            return None
        raise
    tag = str(payload.get("tag_name", "") or "").strip()
    notes_raw = payload.get("body")
    notes = str(notes_raw).strip() if isinstance(notes_raw, str) else None
    if notes and len(notes) > 2000:
        notes = notes[:2000] + "\n…"
    release_url_raw = payload.get("html_url")
    release_url = str(release_url_raw) if isinstance(release_url_raw, str) else None
    available = bool(tag) and _is_newer(tag, __version__)
    return UpdateCheck(
        current_version=__version__,
        latest_version=_normalize_tag(tag) if tag else None,
        update_available=available,
        release_url=release_url,
        release_notes=notes,
        checked_at=now,
    )


def _check_via_commits(slug: str, branch: str, now: float) -> UpdateCheck:
    """Fallback when there are no Releases. Compare HEAD SHAs."""

    payload = _fetch_default_branch_tip(slug, branch)
    remote_sha = str(payload.get("sha", "") or "").strip()
    commit = payload.get("commit") or {}
    msg_raw = commit.get("message") if isinstance(commit, dict) else None
    msg = str(msg_raw).strip() if isinstance(msg_raw, str) else None
    if msg and len(msg) > 1500:
        msg = msg[:1500] + "\n…"
    html_url_raw = payload.get("html_url")
    commit_url = str(html_url_raw) if isinstance(html_url_raw, str) else None

    local_sha = _local_head_sha()
    short_remote = remote_sha[:7] if remote_sha else None
    short_local = local_sha[:7] if local_sha else "unknown"
    label_remote = f"{branch}@{short_remote}" if short_remote else f"{branch}@?"
    available = bool(remote_sha) and bool(local_sha) and (local_sha != remote_sha)
    return UpdateCheck(
        current_version=f"{__version__} ({short_local})",
        latest_version=label_remote,
        update_available=available,
        release_url=commit_url,
        release_notes=msg,
        checked_at=now,
    )


def check_for_update(*, force: bool = False) -> UpdateCheck:
    """Return the latest upstream version vs the running one.

    Tries Releases first (preferred — gives stable version numbers); falls
    back to comparing the local HEAD against the remote default-branch tip
    when the repo has no published releases yet. Cached for
    :data:`_CACHE_TTL_SECONDS` so the React UI can call this on every mount
    without hammering GitHub. Pass ``force=True`` to bypass the cache.
    """

    global _cache_lock_value, _cache_fetched_at
    now = time.time()
    if not force and _cache_lock_value is not None and (now - _cache_fetched_at) < _CACHE_TTL_SECONDS:
        return _cache_lock_value

    slug = _repo_slug()
    branch = _repo_branch()
    try:
        result = _check_via_releases(slug, now)
        if result is None:
            result = _check_via_commits(slug, branch, now)
    except (urlerror.URLError, TimeoutError, ValueError, OSError) as e:
        logger.info("update check failed: %s", e)
        result = UpdateCheck(
            current_version=__version__,
            latest_version=None,
            update_available=False,
            release_url=None,
            release_notes=None,
            checked_at=now,
            error=f"{type(e).__name__}: {e}",
        )

    _cache_lock_value = result
    _cache_fetched_at = now
    return result


def _repo_root() -> Path:
    """The directory containing the install/update scripts."""

    return Path(__file__).resolve().parent.parent


def _update_script() -> Path:
    """Pick the right update script for the current platform."""

    root = _repo_root()
    if os.name == "nt":
        return root / "update.bat"
    return root / "update.sh"


def trigger_update() -> dict[str, str]:
    """Spawn the platform-appropriate update script detached from the API.

    Returns a small status dict — the caller can include it in the HTTP
    response. The script itself logs to ``runs/update.log`` so users can
    tail it after the restart.
    """

    script = _update_script()
    if not script.is_file():
        raise FileNotFoundError(f"Update script not found: {script}")

    log_dir = _repo_root() / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "update.log"
    log_handle = open(log_path, "ab")  # noqa: SIM115 - lifetime spans subprocess

    if os.name == "nt":
        # Detach via DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP so the API
        # process can exit/restart without killing the updater.
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        )
        cmd: list[str] = ["cmd.exe", "/c", str(script)]
        subprocess.Popen(  # noqa: S603 - script path is from the repo root
            cmd,
            cwd=str(_repo_root()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )
    else:
        # POSIX: setsid puts the updater in its own session so SIGHUP from
        # the dying API process doesn't reach it.
        subprocess.Popen(  # noqa: S603 - script path is from the repo root
            ["/bin/sh", str(script)],
            cwd=str(_repo_root()),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    logger.info("update spawned: %s (log: %s)", script, log_path)
    return {
        "spawned": "true",
        "script": str(script),
        "log_path": str(log_path),
        "python": sys.executable,
    }
