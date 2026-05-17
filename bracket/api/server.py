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
    ConfigBundleOut,
    ConfigImportIn,
    FieldSpecOut,
    GalleryGroupOut,
    GalleryItemOut,
    HealthOut,
    JudgeStatusOut,
    LossSeriesOut,
    ModelFamilyOut,
    MonitorSnapshotOut,
    PresetOut,
    PromoteRunRequest,
    PromoteRunResponse,
    ReportOut,
    RunDetailOut,
    RunLogChunkOut,
    StartSessionRequest,
    StartSessionResponse,
    StopSessionResponse,
    TrainingConfigExportOut,
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
from bracket.search.space import SearchOverrides
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


def _read_session_meta(output_dir: Path) -> dict[str, object]:
    """Read ``<output_dir>/session.json`` written by /session/start.

    Returns an empty dict if the file is missing or malformed; callers
    handle the empty case (typically meaning the session was started
    before this metadata was persisted).
    """

    meta_path = output_dir / "session.json"
    if not meta_path.is_file():
        return {}
    try:
        import json
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning("could not read session.json at %s: %s", meta_path, e)
        return {}


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
        grad_norms=list(getattr(ls, "grad_norms", []) or []),
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

    # Subset (or full dataset). ``images_per_dataset <= 0`` is the
    # "full dataset" signal — we still run build_subset() so the
    # user-supplied TOML gets normalised into the sd-scripts schema
    # (musubi-tuner's ``image_directory`` → sd-scripts' nested
    # ``[[datasets.subsets]].image_dir``), but with no per-class cap
    # and no file copy so the trainer reads from the original
    # directories directly. Skipping this step would break sd-scripts
    # any time the source TOML uses musubi naming.
    out = Path(req.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    subset_dir = out / "subset"
    full_dataset = int(req.images_per_dataset) <= 0
    try:
        subset_toml = build_subset(
            source_toml=Path(req.dataset_toml).expanduser(),
            target_dir=subset_dir,
            spec=DatasetSubsetSpec(
                images_per_dataset=0 if full_dataset else int(req.images_per_dataset),
                seed=0,
                copy_files=not full_dataset,
            ),
        )
    except Exception as e:  # noqa: BLE001
        return StartSessionResponse(
            status="bad_request",
            message=f"Dataset subset failed: {type(e).__name__}: {e}",
        )

    sp = Path(req.sample_prompts).expanduser() if req.sample_prompts.strip() else None
    sp_ood = (
        Path(req.sample_prompts_ood).expanduser()
        if (req.sample_prompts_ood or "").strip()
        else None
    )
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
            n_samples=max(1, int(getattr(req, "judge_n_samples", 1) or 1)),
        ))

    clip_iqa = None
    if bool(getattr(req, "enable_clip_iqa_gate", False)) and sp is not None:
        from bracket.judge.clip_iqa import ClipIqaJudge, ClipIqaJudgeConfig
        clip_iqa = ClipIqaJudge(ClipIqaJudgeConfig(
            dq_threshold=float(getattr(req, "clip_iqa_dq_threshold", 0.30) or 0.30),
        ))

    if req.search_method == "optuna":
        controller: SearchController = OptunaTPESearch(
            seed=0, n_startup_trials=int(req.optuna_startup),
        )
    else:
        controller = RandomSearch(seed=0)

    # User-supplied search-range overrides. The dataclass is a no-op when
    # every field is None, so passing it unconditionally is cheap.
    search_overrides = SearchOverrides(
        lr_min=req.lr_min,
        lr_max=req.lr_max,
        batch_size_min=req.batch_size_min,
        batch_size_max=req.batch_size_max,
        gradient_checkpointing_mode=req.gradient_checkpointing_mode,
    )

    def run_fn() -> OrchestrationResult:
        return orchestrate(
            trainer=trainer, dataset_toml=subset_toml, output_dir=out,
            controller=controller, budget_runs=int(req.budget),
            max_steps_per_run=int(req.max_steps),
            max_wall_seconds_per_run=int(req.wall_secs),
            seeds_per_config=int(req.seeds),
            n_curated=int(req.n_curated),
            sample_prompts=sp,
            sample_prompts_ood=sp_ood,
            sample_every_n_steps=sample_every,
            sample_judge=judge,
            clip_iqa_judge=clip_iqa,
            clip_iqa_dq_threshold=float(
                getattr(req, "clip_iqa_dq_threshold", 0.30) or 0.30
            ),
            loss_weight=float(req.judge_loss_weight),
            sample_weight=float(req.judge_sample_weight),
            stop_event=session.stop_event,
            search_overrides=search_overrides,
            use_history_priors=bool(getattr(req, "use_history_priors", False)),
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

    # Persist the full request to <output_dir>/session.json so the
    # promote/export flows survive a server restart (they need the
    # original dataset_toml + sample_prompts paths to build full
    # training runs from search candidates).
    source_toml_path = Path(req.dataset_toml).expanduser().resolve()
    try:
        import json
        (out / "session.json").write_text(
            json.dumps({
                "request": req.model_dump(),
                "source_dataset_toml": str(source_toml_path),
                "subset_dataset_toml": str(subset_toml),
                "started_at": time.time(),
            }, indent=2),
            encoding="utf-8",
        )
    except OSError as e:  # noqa: BLE001 - best effort; never block start
        logger.warning("could not persist session.json: %s", e)

    session.start(
        run_fn, finals_fn=finals_fn, output_dir=out,
        total_runs_target=total_target,
        judge_configured=judge_configured,
        source_dataset_toml=source_toml_path,
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

    @router.get("/runs/{run_id}/log", response_model=RunLogChunkOut)
    def run_log(
        run_id: str,
        offset: int = Query(default=0, ge=0),
        max_bytes: int = Query(default=256 * 1024, ge=1024, le=1024 * 1024),
    ) -> RunLogChunkOut:
        """Return a chunk of the run's stdout.log starting at ``offset``.

        Powers the Monitor page's live console pane. Clients pass back
        ``next_offset`` from the previous response to fetch only the new
        bytes. When the file is larger than ``max_bytes`` from ``offset``,
        the response is capped to the last ``max_bytes`` of the file and
        ``truncated_to_tail`` is set (history older than that is dropped
        for that single poll — the next poll resumes from EOF).
        """

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            return RunLogChunkOut(
                exists=False, content="", offset=offset, next_offset=offset,
                total_size=0, truncated_to_tail=False,
            )
        log_path = Path(out_dir) / "runs" / run_id / "logs" / "stdout.log"
        if not log_path.is_file():
            return RunLogChunkOut(
                exists=False, content="", offset=offset, next_offset=offset,
                total_size=0, truncated_to_tail=False,
            )
        try:
            size = log_path.stat().st_size
        except OSError:
            return RunLogChunkOut(
                exists=False, content="", offset=offset, next_offset=offset,
                total_size=0, truncated_to_tail=False,
            )
        # File was rotated or truncated since last poll — client must reset.
        if offset > size:
            offset = 0
        read_offset = offset
        truncated_to_tail = False
        if size - read_offset > max_bytes:
            read_offset = size - max_bytes
            truncated_to_tail = True
        try:
            with log_path.open("rb") as f:
                f.seek(read_offset)
                chunk = f.read(size - read_offset)
        except OSError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"failed reading log: {e}",
            ) from e
        # `errors="replace"` keeps decoding robust even when the trainer's
        # tqdm output produces stray bytes mid-progress-bar.
        return RunLogChunkOut(
            exists=True,
            content=chunk.decode("utf-8", errors="replace"),
            offset=read_offset,
            next_offset=size,
            total_size=size,
            truncated_to_tail=truncated_to_tail,
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
        return LossSeriesOut(
            steps=list(ls.steps), raw=list(ls.raw), smoothed=list(ls.smoothed),
            grad_norms=list(getattr(ls, "grad_norms", []) or []),
        )

    # ── training-config export / promote ──

    @router.get(
        "/runs/{run_id}/training-config",
        response_model=TrainingConfigExportOut,
    )
    def export_training_config(run_id: str) -> TrainingConfigExportOut:
        """Return everything needed to reproduce or promote a search
        candidate as a full training run.

        Bundles the candidate's hyperparameters with the session-level
        source paths (full ``dataset_toml`` and ``sample_prompts``) and
        the original preset field values. The React UI offers this as a
        per-row "Export" button so users can save the winning config.
        """

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

        meta = _read_session_meta(Path(out_dir))
        request: dict[str, object] = (meta.get("request") or {})  # type: ignore[assignment]
        cfg_raw = match.get("config") or {}
        score_val = match.get("score")
        if isinstance(score_val, str):
            score_val = None
        return TrainingConfigExportOut(
            run_id=str(match.get("run_id", "")),
            role=str(match.get("role", "")),
            family=str(request.get("family", "") or ""),
            training_type=str(request.get("training_type", "") or ""),
            score=float(score_val) if isinstance(score_val, (int, float)) else None,
            config={str(k): str(v) for k, v in cfg_raw.items()},
            source_dataset_toml=str(meta.get("source_dataset_toml") or "") or None,
            sample_prompts=str(request.get("sample_prompts", "") or "") or None,
            preset_field_values={
                str(k): str(v)
                for k, v in (request.get("preset_field_values") or {}).items()
            },
            notes=(
                "Use this with /api/runs/{run_id}/promote to start a full "
                "training run, or paste the values into the Setup tab "
                "manually."
            ),
        )

    # ── config import / export ──

    @router.get("/config", response_model=ConfigBundleOut)
    def export_config() -> ConfigBundleOut:
        """Return the current session's full ``StartSessionRequest`` plus
        a small metadata envelope. The React UI offers this as
        "Export config" so the user can save / share / version-control
        a full bracket setup."""

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No session config saved yet. Start a session first.",
            )
        meta = _read_session_meta(Path(out_dir))
        req_dict = meta.get("request") or {}
        try:
            req = StartSessionRequest(**req_dict)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"session.json is malformed: {type(e).__name__}: {e}",
            ) from e
        return ConfigBundleOut(
            bracket_version=__version__,
            saved_at=float(meta.get("started_at") or time.time()),
            request=req,
        )

    @router.post(
        "/runs/{run_id}/promote",
        response_model=PromoteRunResponse,
        responses={
            409: {"model": PromoteRunResponse},
            400: {"model": PromoteRunResponse},
            404: {"model": PromoteRunResponse},
        },
    )
    def promote_run(run_id: str, req: PromoteRunRequest) -> Response:
        """Start a full training run using a search candidate's
        hyperparameters.

        - Skips the search-time dataset subset; uses the original
          ``dataset_toml`` from the session metadata (or an explicit
          override in the request body).
        - Records the row in the same ledger with ``role="promoted"``.
        - Replaces the active session — only one OrchestrationSession
          runs at a time. Caller should ensure the current session is
          stopped or done before promoting.
        """

        session = get_session()
        if session.is_running():
            return JSONResponse(
                content=PromoteRunResponse(
                    status="conflict",
                    message="A session is already running. Stop it before promoting.",
                ).model_dump(),
                status_code=409,
            )

        out_dir = session.snapshot().output_dir
        if out_dir is None:
            return JSONResponse(
                content=PromoteRunResponse(
                    status="bad_request",
                    message="No session loaded — cannot resolve the source run.",
                ).model_dump(),
                status_code=400,
            )

        rows = read_ledger(Path(out_dir) / "ledger.jsonl")
        source_row = next((r for r in rows if r.get("run_id") == run_id), None)
        if source_row is None:
            return JSONResponse(
                content=PromoteRunResponse(
                    status="bad_request",
                    message=f"Unknown source run: {run_id}",
                ).model_dump(),
                status_code=404,
            )

        meta = _read_session_meta(Path(out_dir))
        request_dict: dict[str, object] = (meta.get("request") or {})  # type: ignore[assignment]
        if not request_dict:
            return JSONResponse(
                content=PromoteRunResponse(
                    status="bad_request",
                    message=(
                        "session.json is missing — cannot resolve trainer / "
                        "weights / source dataset. Start the original session "
                        "from a recent Bracket build first."
                    ),
                ).model_dump(),
                status_code=400,
            )

        try:
            orig_req = StartSessionRequest(**request_dict)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                content=PromoteRunResponse(
                    status="bad_request",
                    message=f"session.json is malformed: {type(e).__name__}: {e}",
                ).model_dump(),
                status_code=400,
            )

        preset = get_preset(orig_req.family, orig_req.training_type)
        if preset is None:
            return JSONResponse(
                content=PromoteRunResponse(
                    status="bad_request",
                    message=f"Unknown preset: {orig_req.family} / {orig_req.training_type}",
                ).model_dump(),
                status_code=400,
            )

        try:
            trainer = preset.trainer_factory(
                **orig_req.preset_field_values,
                vram_gb=float(orig_req.vram_gb) if orig_req.vram_gb else None,
            )
        except Exception as e:  # noqa: BLE001
            return JSONResponse(
                content=PromoteRunResponse(
                    status="bad_request",
                    message=f"Trainer construction failed: {type(e).__name__}: {e}",
                ).model_dump(),
                status_code=400,
            )

        # Source dataset: explicit override > session metadata source toml.
        if req.full_dataset_toml.strip():
            source_toml = Path(req.full_dataset_toml).expanduser()
        else:
            source_toml = Path(str(meta.get("source_dataset_toml") or "")).expanduser()
        if not source_toml.is_file():
            return JSONResponse(
                content=PromoteRunResponse(
                    status="bad_request",
                    message=f"Source dataset_toml not found: {source_toml}",
                ).model_dump(),
                status_code=400,
            )

        sp_str = (orig_req.sample_prompts or "").strip()
        sp = Path(sp_str).expanduser() if sp_str else None
        judge = None
        if orig_req.judge_method == "lmstudio" and sp is not None:
            from bracket.judge.lmstudio import LMStudioJudge as _LMStudioJudge
            from bracket.judge.lmstudio import LMStudioJudgeConfig as _LMStudioJudgeConfig
            enable_thinking = False if orig_req.judge_disable_thinking else None
            judge = _LMStudioJudge(_LMStudioJudgeConfig(
                base_url=orig_req.judge_base_url, model=orig_req.judge_model,
                enable_thinking=enable_thinking,
            ))

        # Output directory: explicit override (new) or session output_dir
        # (append the promoted run inside the same session for ledger continuity).
        promoted_out = (
            Path(req.output_dir).expanduser().resolve()
            if req.output_dir else Path(out_dir)
        )
        promoted_out.mkdir(parents=True, exist_ok=True)

        resume = Path(req.resume_from).expanduser() if req.resume_from.strip() else None
        cfg_dict = dict(source_row.get("config") or {})
        max_steps = int(req.max_steps) if int(req.max_steps) > 0 else 2000

        from bracket.orchestrator.promote import run_promoted_for_session

        def promote_fn() -> OrchestrationResult:
            return run_promoted_for_session(
                trainer=trainer,
                source_config=cfg_dict,
                dataset_toml=source_toml,
                output_dir=promoted_out,
                max_steps=max_steps,
                save_every_n_steps=max(1, int(req.save_every_n_steps)),
                save_state=bool(req.save_state),
                resume_from=resume,
                sample_prompts=sp,
                sample_every_n_steps=req.sample_every_n_steps,
                sample_judge=judge,
                loss_weight=float(orig_req.judge_loss_weight),
                sample_weight=float(orig_req.judge_sample_weight),
                base_run_id=run_id,
            )

        session.start(
            promote_fn,
            output_dir=promoted_out,
            total_runs_target=1,
            judge_configured=judge is not None,
            source_dataset_toml=source_toml,
        )
        return JSONResponse(
            content=PromoteRunResponse(
                status="started",
                message=f"Promoted run started from {run_id}.",
                output_dir=str(promoted_out),
            ).model_dump(),
            status_code=200,
        )

    @router.post("/config/validate", response_model=ConfigBundleOut)
    def validate_config(body: ConfigImportIn) -> ConfigBundleOut:
        """Pydantic-validate an imported config bundle.

        The UI can validate client-side via the OpenAPI types but this
        endpoint is the source of truth — surfaces field-level errors
        the same way the start endpoint would.
        """

        return ConfigBundleOut(
            bracket_version=__version__,
            saved_at=time.time(),
            request=body.request,
        )

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
        """Current report.md content as text/markdown.

        Auto-regenerates if ``ledger.jsonl`` is newer than ``report.md``
        (or if no report exists yet but a ledger does) — otherwise the
        Results page sits showing "No runs in ledger" until the user
        clicks Regenerate manually.
        """

        out_dir = get_session().snapshot().output_dir
        if out_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No session has been started yet.",
            )
        out = Path(out_dir)
        ledger_path = out / "ledger.jsonl"
        report_path = out / "report.md"

        report_mtime = report_path.stat().st_mtime if report_path.exists() else 0.0
        ledger_mtime = ledger_path.stat().st_mtime if ledger_path.exists() else 0.0
        if ledger_path.exists() and ledger_mtime > report_mtime:
            try:
                generate_report(ledger_path, report_path)
            except Exception:
                logger.exception("auto-regenerating report failed; serving stale copy")

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
