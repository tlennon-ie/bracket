"""FastAPI app factory + routers for the React frontend.

Every endpoint here mirrors a Gradio event handler in ``bracket/ui/app.py``
so the frontend migration plan documented in
``docs/FRONTEND_MIGRATION_PLAN.md`` §3 can be implemented without touching
the orchestrator.

Threading model
---------------
A single module-level :class:`OrchestrationSession` is shared with the
Gradio app. Both UIs may run side-by-side; the session is the single source
of truth and is already RLock-thread-safe.

Security
--------
The static-file route ``/files/{run_id}/{rel_path}`` does symlink resolution
and verifies the result stays under ``output_dir/runs``; this is the only
piece that touches user-supplied paths after request parsing. Treated as if
the API were going to be exposed to the open internet — the muscle is right
even though v0.1 is single-user local.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from bracket import __version__
from bracket.api.schemas import (
    CandidateRowOut,
    FieldSpecOut,
    GalleryGroupOut,
    GalleryItemOut,
    HealthOut,
    JudgeStatusOut,
    LossSeriesOut,
    ModelFamilyOut,
    MonitorSnapshotOut,
    PresetOut,
    ReportOut,
    RunDetailOut,
    StartSessionRequest,
    StartSessionResponse,
    StopSessionResponse,
    TrainingTypeOut,
    TriggerUpdateOut,
    UpdateStatusOut,
)
from bracket.dataset.subset import DatasetSubsetSpec, build_subset
from bracket.judge.lmstudio import LMStudioJudge, LMStudioJudgeConfig
from bracket.orchestrator.finals import run_finals_stage
from bracket.orchestrator.loop import OrchestrationResult, orchestrate
from bracket.proof.report import generate_report
from bracket.registry import (
    PRESETS,
    SESSION_FIELDS,
    FieldSpec,
    ModelPreset,
    get_preset,
    list_model_families,
    training_types_for,
)
from bracket.search.controller import RandomSearch, SearchController
from bracket import updater as bracket_updater
from bracket.search.optuna_search import OptunaTPESearch
from bracket.ui.monitor import (
    MonitorSnapshot,
    build_snapshot,
    find_tfevents_in,
    gallery_groups,
    load_loss_series,
    read_ledger,
    score_history_rows,
)
from bracket.ui.session import OrchestrationSession, SessionState

logger = logging.getLogger("bracket.api")


# ───────────────────────────── singleton session ─────────────────────────────

# Shared with the Gradio app so both surfaces drive the same orchestrator.
# The Gradio app lazily imports its own copy; we expose ours here so test
# fixtures can swap in a fresh one via ``get_session`` dependency override.
_SESSION: OrchestrationSession = OrchestrationSession()


def get_session() -> OrchestrationSession:
    """Dependency-injection seam. Tests override this to isolate sessions."""

    return _SESSION


# ───────────────────────────── helpers ─────────────────────────────


_VALID_FILE_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".webp",
    ".mp4", ".webm", ".mov", ".mkv",
    ".txt", ".log", ".md", ".json",
})


def _field_spec_to_out(f: FieldSpec) -> FieldSpecOut:
    return FieldSpecOut(
        name=f.name, label=f.label, default=f.default, required=f.required,
        kind=f.kind, help=f.help, target=f.target,
    )


def _preset_to_out(p: ModelPreset) -> PresetOut:
    return PresetOut(
        id=p.id, family=p.model_family, training_type=p.training_type,
        display_name=p.display_name, notes=p.notes,
        needs_pre_cache=p.needs_pre_cache,
        fields=[_field_spec_to_out(f) for f in p.fields],
        session_fields=[_field_spec_to_out(f) for f in SESSION_FIELDS],
    )


def _candidate_row_to_out(r: object) -> CandidateRowOut:
    """Convert a :class:`bracket.ui.monitor.CandidateRow` dataclass to schema."""

    return CandidateRowOut(
        run_id=getattr(r, "run_id", ""),
        role=getattr(r, "role", ""),
        config_id=getattr(r, "config_id", ""),
        score=getattr(r, "score", None),
        final_smoothed=getattr(r, "final_smoothed", None),
        slope=getattr(r, "slope", None),
        n_steps=int(getattr(r, "n_steps", 0) or 0),
        duration_s=float(getattr(r, "duration_s", 0.0) or 0.0),
        disqualified=getattr(r, "disqualified", None),
    )


def _loss_series_to_out(ls: object) -> Optional[LossSeriesOut]:
    if ls is None:
        return None
    return LossSeriesOut(
        steps=list(getattr(ls, "steps", []) or []),
        raw=list(getattr(ls, "raw", []) or []),
        smoothed=list(getattr(ls, "smoothed", []) or []),
    )


def _build_snapshot_payload(
    session: OrchestrationSession, ema_alpha: float = 0.05,
) -> MonitorSnapshotOut:
    """Produce a :class:`MonitorSnapshotOut` for the current session state.

    Returns a coherent idle snapshot when no session has been started yet.
    """

    sess_snap: SessionState = session.snapshot()
    out_dir = sess_snap.output_dir
    if out_dir is None:
        return MonitorSnapshotOut(
            session_status=sess_snap.status, output_dir=None,
            elapsed_s=sess_snap.elapsed_s(),
            error_message=sess_snap.error_message,
            status_line="_idle / no session yet_",
            ts=time.time(),
        )

    snap: MonitorSnapshot = build_snapshot(
        out_dir,
        total_runs_target=sess_snap.total_runs_target,
        ema_alpha=ema_alpha,
        judge_configured=sess_snap.judge_configured,
    )
    return MonitorSnapshotOut(
        session_status=sess_snap.status,
        output_dir=str(out_dir),
        elapsed_s=sess_snap.elapsed_s(),
        error_message=sess_snap.error_message,
        status_line=snap.status_line,
        progress_pct=float(snap.progress_pct),
        completed_runs=int(snap.completed_runs),
        total_runs_target=int(snap.total_runs_target),
        current_run_id=snap.current_run_id,
        current_run_steps_done=snap.current_run_steps_done,
        current_run_max_steps=snap.current_run_max_steps,
        current_loss=_loss_series_to_out(snap.current_loss),
        score_history=[_candidate_row_to_out(r) for r in snap.score_history],
        setup_status=snap.setup_status,
        judge_summary=snap.judge_summary,
        session_done=bool(snap.session_done),
        current_steps_per_sec=snap.current_steps_per_sec,
        ts=time.time(),
    )


def _gallery_groups_to_out(out_dir: Path) -> list[GalleryGroupOut]:
    """Wrap :func:`bracket.ui.monitor.gallery_groups` and rewrite filesystem
    paths into HTTP URLs that the static-file mount serves."""

    groups = gallery_groups(out_dir)
    api_groups: list[GalleryGroupOut] = []
    for g in groups:
        items: list[GalleryItemOut] = []
        for path, name in g.items:
            try:
                rel = Path(path).resolve().relative_to(
                    (Path(out_dir).resolve() / "runs" / g.run_id)
                )
                url = f"/files/{g.run_id}/{rel.as_posix()}"
            except ValueError:
                # Path lay outside the expected run dir — skip rather than
                # advertise a URL the static mount will refuse.
                continue
            items.append(GalleryItemOut(
                path=str(path), url=url, caption=name, run_id=g.run_id,
            ))
        api_groups.append(GalleryGroupOut(
            run_id=g.run_id, items=items, mtime=float(g.mtime),
        ))
    return api_groups


# ───────────────────────────── start-session core ─────────────────────────────


def _start_session_impl(
    req: StartSessionRequest, session: OrchestrationSession,
) -> StartSessionResponse:
    """Pure-Python core shared with (a future refactor of) the Gradio path.

    Mirrors ``bracket/ui/app.py::_start_session`` exactly: same validation,
    same trainer construction, same orchestrate() invocation in a background
    thread via ``OrchestrationSession.start``.

    Returns a :class:`StartSessionResponse`. Callers translate ``status`` to
    the appropriate HTTP code.
    """

    if session.is_running():
        return StartSessionResponse(
            status="conflict",
            message="A session is already running.",
        )

    preset = get_preset(req.family, req.training_type)
    if preset is None:
        return StartSessionResponse(
            status="bad_request",
            message=f"Unknown preset: {req.family} / {req.training_type}",
        )

    fields = list(preset.fields)
    field_values: dict[str, str] = {
        f.name: (req.preset_field_values.get(f.name, "") or "").strip()
        for f in fields
    }

    # Validate required preset fields + the two required session fields.
    missing: list[str] = [f.label for f in fields if f.required and not field_values.get(f.name)]
    if not req.dataset_toml.strip():
        missing.append("Dataset config TOML *")
    if not req.output_dir.strip():
        missing.append("Session output directory *")
    if missing:
        return StartSessionResponse(
            status="bad_request",
            message="Missing required fields: " + ", ".join(missing),
        )

    # Build trainer via the preset factory.
    try:
        trainer = preset.trainer_factory(
            **field_values,
            vram_gb=float(req.vram_gb) if req.vram_gb else None,
        )
    except Exception as e:  # noqa: BLE001
        return StartSessionResponse(
            status="bad_request",
            message=f"Trainer construction failed: {type(e).__name__}: {e}",
        )

    # Subset.
    out = Path(req.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    subset_dir = out / "subset"
    try:
        subset_toml = build_subset(
            source_toml=Path(req.dataset_toml).expanduser(),
            target_dir=subset_dir,
            spec=DatasetSubsetSpec(images_per_dataset=int(req.images_per_dataset), seed=0),
        )
    except Exception as e:  # noqa: BLE001
        return StartSessionResponse(
            status="bad_request",
            message=f"Dataset subset failed: {type(e).__name__}: {e}",
        )

    sp = Path(req.sample_prompts).expanduser() if req.sample_prompts.strip() else None
    sample_every = int(req.max_steps) if sp is not None else None

    judge = None
    if req.judge_method == "lmstudio" and sp is not None:
        # `judge_disable_thinking=True` translates to enable_thinking=False
        # in the chat_template_kwargs sent to LMStudio. Leave None when the
        # user wants the model's default behaviour.
        enable_thinking = False if req.judge_disable_thinking else None
        judge = LMStudioJudge(LMStudioJudgeConfig(
            base_url=req.judge_base_url, model=req.judge_model,
            enable_thinking=enable_thinking,
        ))

    if req.search_method == "optuna":
        controller: SearchController = OptunaTPESearch(
            seed=0, n_startup_trials=int(req.optuna_startup),
        )
    else:
        controller = RandomSearch(seed=0)

    def run_fn() -> OrchestrationResult:
        return orchestrate(
            trainer=trainer, dataset_toml=subset_toml, output_dir=out,
            controller=controller, budget_runs=int(req.budget),
            max_steps_per_run=int(req.max_steps),
            max_wall_seconds_per_run=int(req.wall_secs),
            seeds_per_config=int(req.seeds),
            n_curated=int(req.n_curated),
            sample_prompts=sp, sample_every_n_steps=sample_every,
            sample_judge=judge,
            loss_weight=float(req.judge_loss_weight),
            sample_weight=float(req.judge_sample_weight),
            stop_event=session.stop_event,
        )

    finals_fn = None
    if int(req.finals_top_k) > 0:
        def finals_fn(stage1: OrchestrationResult) -> None:  # noqa: F811
            run_finals_stage(
                trainer=trainer, stage1=stage1, dataset_toml=subset_toml,
                output_dir=out, top_k=int(req.finals_top_k),
                finals_max_steps=int(req.finals_max_steps),
                finals_max_wall_seconds=int(int(req.finals_max_steps) * 12),
                finals_seeds_per_config=int(req.finals_seeds),
                sample_prompts=sp, sample_every_n_steps=sample_every,
                sample_judge=judge,
                loss_weight=float(req.judge_loss_weight),
                sample_weight=float(req.judge_sample_weight),
            )

    total_target = (1 + int(req.budget)) * int(req.seeds)
    # Judge intent flag — drives the "configured but no scored row yet"
    # snapshot message so users who Stop early don't see a misleading
    # "not configured" warning.
    judge_configured = judge is not None and sp is not None
    session.start(
        run_fn, finals_fn=finals_fn, output_dir=out,
        total_runs_target=total_target,
        judge_configured=judge_configured,
    )

    return StartSessionResponse(
        status="started",
        message=f"Started. Output: {out}",
        session_id=out.name,
        output_dir=str(out),
    )


# ───────────────────────────── routers ─────────────────────────────


def _make_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthOut)
    def health() -> HealthOut:
        """Liveness probe. Replaces no Gradio handler — new endpoint."""

        return HealthOut(status="ok", version=__version__)

    # ── presets ──

    @router.get("/presets/families", response_model=list[ModelFamilyOut])
    def list_families() -> list[ModelFamilyOut]:
        """List model families. Powers the first cascading dropdown.

        Replaces the construction of ``family_dd`` choices in
        ``bracket/ui/app.py::build_app``.
        """

        return [ModelFamilyOut(name=fam, label=fam) for fam in list_model_families()]

    @router.get("/presets/families/{family}/types", response_model=list[TrainingTypeOut])
    def list_types(family: str) -> list[TrainingTypeOut]:
        """List training types for a family. Replaces ``_on_family_change``."""

        types = training_types_for(family)
        if not types:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown model family: {family}",
            )
        return [TrainingTypeOut(name=t, label=t) for t in types]

    @router.get("/presets/{family}/{training_type}", response_model=PresetOut)
    def get_preset_spec(family: str, training_type: str) -> PresetOut:
        """Return the preset for a chosen pair. Replaces ``_preset_fields``."""

        preset = get_preset(family, training_type)
        if preset is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown preset: {family} / {training_type}",
            )
        return _preset_to_out(preset)

    # ── session ──

    @router.get("/session", response_model=MonitorSnapshotOut)
    def session_snapshot() -> MonitorSnapshotOut:
        """One-shot session snapshot. Replaces ``_monitor_refresh`` data feed."""

        return _build_snapshot_payload(get_session())

    @router.post(
        "/session/start", response_model=StartSessionResponse,
        responses={
            409: {"model": StartSessionResponse},
            400: {"model": StartSessionResponse},
        },
    )
    def session_start(req: StartSessionRequest) -> Response:
        """Start an orchestration session. Replaces ``_start_session``."""

        result = _start_session_impl(req, get_session())
        if result.status == "started":
            return JSONResponse(content=result.model_dump(), status_code=200)
        if result.status == "conflict":
            return JSONResponse(content=result.model_dump(), status_code=409)
        return JSONResponse(content=result.model_dump(), status_code=400)

    @router.post("/session/stop", response_model=StopSessionResponse)
    def session_stop() -> StopSessionResponse:
        """Stop the running session. Replaces ``_stop_session``."""

        sess = get_session()
        if not sess.is_running():
            return StopSessionResponse(stopped=False, message="Nothing running.")
        sess.stop()
        return StopSessionResponse(
            stopped=True,
            message="Stop requested. Killing current candidate; loop will exit.",
        )

    # ── runs ──

    @router.get("/runs", response_model=list[CandidateRowOut])
    def list_runs() -> list[CandidateRowOut]:
        """Score history. Replaces the ``history_table`` data feed."""

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            return []
        rows = read_ledger(Path(out_dir) / "ledger.jsonl")
        return [_candidate_row_to_out(r) for r in score_history_rows(rows)]

    @router.get("/runs/{run_id}", response_model=RunDetailOut)
    def run_detail(run_id: str) -> RunDetailOut:
        """Per-run detail. Replaces a future click-into-row UX."""

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No session has been started yet.",
            )
        rows = read_ledger(Path(out_dir) / "ledger.jsonl")
        match = next((r for r in rows if r.get("run_id") == run_id), None)
        if match is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown run_id: {run_id}",
            )
        run_dir = Path(out_dir) / "runs" / run_id
        log_path = run_dir / "logs" / "stdout.log"
        sample_dir = run_dir / "output" / "sample"
        cfg_raw = match.get("config") or {}
        comps_raw = match.get("score_components") or {}
        judge_raw = match.get("judge_report") or {}
        score_val = match.get("score")
        if isinstance(score_val, str):
            score_val = None
        return RunDetailOut(
            run_id=str(match.get("run_id", "")),
            role=str(match.get("role", "")),
            config_id=str(match.get("config_id", "")),
            score=float(score_val) if isinstance(score_val, (int, float)) else None,
            n_steps=int(match.get("n_steps", 0) or 0),
            duration_s=float(match.get("duration_s", 0.0) or 0.0),
            disqualified=match.get("disqualified"),
            config={str(k): str(v) for k, v in cfg_raw.items()},
            score_components={
                str(k): float(v) for k, v in comps_raw.items()
                if isinstance(v, (int, float))
            },
            judge_report={
                str(k): float(v) for k, v in judge_raw.items()
                if isinstance(v, (int, float))
            },
            run_dir=str(run_dir),
            log_path=str(log_path) if log_path.exists() else None,
            sample_dir=str(sample_dir) if sample_dir.exists() else None,
        )

    @router.get("/runs/{run_id}/loss", response_model=LossSeriesOut)
    def run_loss(
        run_id: str,
        ema_alpha: float = Query(default=0.05, ge=0.001, le=1.0),
    ) -> LossSeriesOut:
        """Loss series for one run. Powers the live loss chart."""

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No session has been started yet.",
            )
        run_dir = Path(out_dir) / "runs" / run_id
        if not run_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown run_id: {run_id}",
            )
        tfe = find_tfevents_in(run_dir)
        if tfe is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No tfevents file under run {run_id}",
            )
        ls = load_loss_series(tfe, ema_alpha=ema_alpha)
        if ls is None:
            return LossSeriesOut()
        return LossSeriesOut(steps=list(ls.steps), raw=list(ls.raw), smoothed=list(ls.smoothed))

    # ── gallery ──

    @router.get("/gallery", response_model=list[GalleryGroupOut])
    def gallery() -> list[GalleryGroupOut]:
        """Sample-image gallery, grouped by run. Replaces ``_flat_gallery_items``."""

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            return []
        return _gallery_groups_to_out(Path(out_dir))

    # ── report ──

    @router.post("/report/regenerate", response_model=ReportOut)
    def report_regenerate() -> ReportOut:
        """Regenerate the markdown report. Replaces ``_results_refresh``'s
        report-rebuild path."""

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No session has been started yet.",
            )
        out = Path(out_dir)
        ledger = out / "ledger.jsonl"
        report_path = out / "report.md"
        if not ledger.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No ledger yet — nothing to report.",
            )
        try:
            generate_report(ledger, report_path)
        except Exception as e:  # noqa: BLE001
            logger.exception("report regeneration failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Report generation failed: {type(e).__name__}: {e}",
            ) from e
        return ReportOut(
            path=str(report_path),
            content=report_path.read_text(encoding="utf-8") if report_path.exists() else "",
        )

    @router.get("/report")
    def report_get() -> Response:
        """Current report.md content as text/markdown."""

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No session has been started yet.",
            )
        report_path = Path(out_dir) / "report.md"
        if not report_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="report.md has not been generated yet.",
            )
        return PlainTextResponse(
            content=report_path.read_text(encoding="utf-8"),
            media_type="text/markdown",
        )

    # ── judge ──

    @router.get("/judge/status", response_model=JudgeStatusOut)
    def judge_status() -> JudgeStatusOut:
        """Judge configuration status. Derived from the session snapshot."""

        snap = _build_snapshot_payload(get_session())
        summary = snap.judge_summary or "Judge: not yet evaluated."
        # The build_snapshot helper marks "not configured" via the leading
        # cross-mark; otherwise it reports counts.
        configured = bool(summary) and "not configured" not in summary
        return JudgeStatusOut(configured=configured, summary=summary)

    # ── updater ──

    @router.get("/update/check", response_model=UpdateStatusOut)
    def update_check(force: bool = Query(default=False)) -> UpdateStatusOut:
        """Compare the running version against the latest GitHub release.

        Cached for 30 minutes to avoid hammering GitHub's anonymous rate
        limit. The frontend polls this on mount and surfaces a toast when
        ``update_available`` is True.
        """

        result = bracket_updater.check_for_update(force=force)
        return UpdateStatusOut(
            current_version=result.current_version,
            latest_version=result.latest_version,
            update_available=result.update_available,
            release_url=result.release_url,
            release_notes=result.release_notes,
            checked_at=result.checked_at,
            error=result.error,
        )

    @router.post("/update/apply", response_model=TriggerUpdateOut)
    def update_apply() -> TriggerUpdateOut:
        """Spawn the platform-appropriate update script and return.

        The script does ``git pull``, rebuilds the venv + frontend, and
        re-launches the server. It runs detached, so this endpoint returns
        successfully even though the API process will be replaced shortly
        after. The React frontend should expect the WebSocket to drop and
        reconnect once the new server is up.
        """

        try:
            result = bracket_updater.trigger_update()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e),
            ) from e
        except OSError as e:
            logger.exception("update spawn failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to spawn update script: {type(e).__name__}: {e}",
            ) from e
        return TriggerUpdateOut(
            spawned=True,
            message="Update script started. The server will restart shortly.",
            script=result.get("script"),
            log_path=result.get("log_path"),
        )

    return router


# ───────────────────────────── websocket ─────────────────────────────


async def _ws_snapshot_loop(websocket: WebSocket) -> None:
    """Push a snapshot on a 1 s cadence while running, 5 s when idle.

    Uses ``asyncio.to_thread`` so the synchronous disk-read in
    ``build_snapshot`` doesn't block the event loop.
    """

    sess = get_session()
    while True:
        try:
            payload: MonitorSnapshotOut = await asyncio.to_thread(
                _build_snapshot_payload, sess,
            )
        except Exception:
            logger.exception("snapshot build failed")
            await asyncio.sleep(2.0)
            continue
        try:
            await websocket.send_json(payload.model_dump())
        except WebSocketDisconnect:
            return
        cadence = 1.0 if payload.session_status in ("running", "stopping") else 5.0
        await asyncio.sleep(cadence)


# ───────────────────────────── static files ─────────────────────────────


def _make_files_router() -> APIRouter:
    router = APIRouter()

    @router.get("/files/{run_id}/{rel_path:path}")
    def serve_run_file(run_id: str, rel_path: str) -> FileResponse:
        """Serve a sample/log file under ``<output_dir>/runs/<run_id>/...``.

        Hardened against path traversal:
          1. Reject ``..`` and absolute markers up front.
          2. Resolve symlinks; verify the final path is inside
             ``output_dir/runs``.
          3. Whitelist file extensions — never serve weights.
        """

        # Step 1: cheap textual rejects.
        if ".." in rel_path.split("/") or rel_path.startswith(("/", "\\")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Path traversal rejected.",
            )
        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No session has been started yet.",
            )
        runs_root = Path(out_dir).resolve() / "runs"

        # Step 2: resolve and verify containment.
        candidate = (runs_root / run_id / rel_path)
        try:
            resolved = candidate.resolve(strict=False)
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Bad path: {e}",
            ) from e
        try:
            resolved.relative_to(runs_root)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Path escapes runs directory.",
            ) from e

        # Step 3: extension whitelist.
        if resolved.suffix.lower() not in _VALID_FILE_SUFFIXES:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"File extension not whitelisted: {resolved.suffix}",
            )

        if not resolved.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Not found: {run_id}/{rel_path}",
            )
        return FileResponse(
            path=str(resolved),
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    return router


# ───────────────────────────── factory + cli entry ─────────────────────────────


def _resolve_cors_origins(override: Optional[str] = None) -> list[str]:
    raw = override if override is not None else os.environ.get("BRACKET_CORS_ORIGINS", "")
    if not raw.strip():
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app(*, cors_origins: Optional[list[str]] = None) -> FastAPI:
    """FastAPI app factory.

    The default CORS allowlist is the Vite dev server. In production the
    React bundle is served from this same FastAPI process, so no CORS is
    needed; configure ``BRACKET_CORS_ORIGINS=`` to disable the allowlist.
    """

    app = FastAPI(
        title="Bracket API",
        version=__version__,
        description=(
            "REST + WebSocket surface backing the React frontend. "
            "Mirrors every Gradio handler in bracket/ui/app.py."
        ),
    )
    origins = cors_origins if cors_origins is not None else _resolve_cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(_make_router())
    app.include_router(_make_files_router())

    @app.websocket("/api/ws/snapshot")
    async def ws_snapshot(websocket: WebSocket) -> None:
        """Push a :class:`MonitorSnapshotOut` every 1 s while the session is
        running, every 5 s when idle. The client owns reconnection.
        """

        await websocket.accept()
        try:
            await _ws_snapshot_loop(websocket)
        except WebSocketDisconnect:
            return
        except Exception:
            logger.exception("websocket loop crashed")
            try:
                await websocket.close()
            except Exception:
                pass

    # Static React frontend. After ``cd frontend && npm run build``, the SPA
    # bundle lives at ``frontend/dist/``. Mounting last so /api routes win.
    # ``html=True`` makes StaticFiles fall back to index.html for unknown
    # paths so client-side routing works.
    _mount_react_dist(app)

    return app


def _mount_react_dist(app: FastAPI) -> None:
    """Mount the built React SPA at ``/`` if its dist directory exists.

    Resolves ``frontend/dist`` relative to the repo root. If the bundle
    hasn't been built yet, log a hint and continue — the API still serves.

    Two routes are wired:

    1. ``GET /assets/*`` and individual top-level files (``/logo.png`` etc.)
       are served directly from disk via :class:`StaticFiles`.
    2. A catch-all ``GET /{path:path}`` falls back to ``index.html`` so
       client-side routing works for ``/setup``, ``/run``, ``/monitor``,
       ``/results``, etc. ``/api/*``, ``/files/*``, and the websocket are
       declared above this mount and therefore win the route match.
    """

    from fastapi.responses import FileResponse, Response
    from fastapi.staticfiles import StaticFiles

    repo_root = Path(__file__).resolve().parent.parent.parent
    dist = repo_root / "frontend" / "dist"
    if not dist.is_dir() or not (dist / "index.html").is_file():
        logger.info(
            "React bundle not built (no %s). Run 'cd frontend && npm run build'. "
            "API endpoints still work.", dist,
        )
        return

    index_path = dist / "index.html"
    dist_root_resolved = dist.resolve()
    assets_dir = dist / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    async def _spa_root() -> FileResponse:
        return FileResponse(str(index_path), media_type="text/html")

    @app.get("/{spa_path:path}", include_in_schema=False)
    async def _spa_fallback(spa_path: str) -> Response:
        # Reserved paths handled by other mounts/routers. Belt-and-braces
        # since /api, /files, /assets, and the websocket are registered first.
        if spa_path.startswith(("api/", "files/", "assets/")) or spa_path == "ws":
            return Response(status_code=404)

        # Serve real files at the dist root (favicon.svg, logo.png, etc.).
        candidate = (dist / spa_path).resolve()
        try:
            candidate.relative_to(dist_root_resolved)
        except ValueError:
            return Response(status_code=404)
        if candidate.is_file():
            return FileResponse(str(candidate))
        # SPA route — serve index.html so the client router takes over.
        return FileResponse(str(index_path), media_type="text/html")

    logger.info("Mounted React frontend from %s", dist)


def serve(
    *, host: str = "127.0.0.1", port: int = 8000,
    reload: bool = False,
    cors_origins: Optional[list[str]] = None,
) -> None:
    """Synchronous entry point — ``uvicorn.run(create_app(), ...)``."""

    import uvicorn

    if reload:
        # Reload mode requires an import string; fall back to a fixed factory.
        os.environ.setdefault(
            "BRACKET_CORS_ORIGINS",
            ",".join(cors_origins) if cors_origins else "",
        )
        uvicorn.run(
            "bracket.api.server:create_app",
            host=host, port=port, factory=True, reload=True,
        )
        return

    app = create_app(cors_origins=cors_origins)
    uvicorn.run(app, host=host, port=port)
