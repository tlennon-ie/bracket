"""Pydantic schemas for the React frontend.

Every shape here mirrors a concrete dataclass or function-return-type from
the Gradio app so the contract documented in
``docs/FRONTEND_MIGRATION_PLAN.md`` §3 can be type-checked end-to-end.

Conventions:
  - No ``Any`` in the public surface. Every field is typed.
  - Optionals use ``Optional[X]`` rather than ``X | None`` for backwards-compat
    with Python 3.10 deferred-evaluation type-checkers (mypy --strict).
  - All schemas inherit from ``BaseModel`` directly; ``model_config`` is
    declared once on a shared base so the React side gets stable JSON.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ───────────────────────────── base ─────────────────────────────


class _Base(BaseModel):
    """Shared base. ``populate_by_name`` lets us alias snake_case fields if
    the React side ever wants camelCase, without touching the Python types."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ───────────────────────────── presets / fields ─────────────────────────────


class FieldSpecOut(_Base):
    """Mirror of :class:`bracket.registry.FieldSpec`."""

    name: str
    label: str
    default: str = ""
    required: bool = True
    kind: str = "string"
    help: str = ""
    target: str = "trainer"


class ModelFamilyOut(_Base):
    """One entry in the family dropdown."""

    name: str
    label: str


class TrainingTypeOut(_Base):
    """One entry in the training-type dropdown for a chosen family."""

    name: str
    label: str


class PresetOut(_Base):
    """The preset spec returned for a chosen (family, training_type) pair.

    The frontend uses ``fields`` to render the preset-specific input rows
    and ``session_fields`` to render the always-visible session inputs.
    ``notes`` is markdown for the help panel below the cascade.
    """

    id: str
    family: str
    training_type: str
    display_name: str
    notes: str = ""
    needs_pre_cache: bool = False
    fields: list[FieldSpecOut] = Field(default_factory=list)
    session_fields: list[FieldSpecOut] = Field(default_factory=list)


# ───────────────────────────── start session request ─────────────────────────────


class StartSessionRequest(_Base):
    """Body of ``POST /api/session/start``.

    Mirrors the positional parameter list of ``_start_session`` in
    ``bracket/ui/app.py`` exactly. ``preset_field_values`` is a dict keyed
    by FieldSpec.name (rather than the positional Gradio textbox values)
    because the API client knows the field names from ``GET /api/presets``.
    """

    family: str
    training_type: str

    # Session-wide
    dataset_toml: str
    output_dir: str
    sample_prompts: str = ""
    resume: str = ""
    images_per_dataset: int = 12
    vram_gb: Optional[float] = None

    # Search budget
    budget: int = 8
    max_steps: int = 300
    wall_secs: int = 1800
    seeds: int = 1
    search_method: str = "optuna"
    optuna_startup: int = 5
    n_curated: int = -1

    # Finals
    finals_top_k: int = 0
    finals_max_steps: int = 2000
    finals_seeds: int = 2

    # Judge
    judge_method: str = "none"
    judge_base_url: str = "http://localhost:1234/v1"
    judge_model: str = "qwen3-vl-8b-thinking-abliterated"
    judge_loss_weight: float = 0.3
    judge_sample_weight: float = 0.7
    # When True, send `chat_template_kwargs={"enable_thinking": false}`
    # to LMStudio so Qwen3-class VLMs skip the <think>...</think>
    # preamble. Roughly halves judge latency on thinking models. None
    # equivalent (= leave default) is encoded as False here so the field
    # survives JSON round-trip cleanly.
    judge_disable_thinking: bool = False

    # Preset-specific values, keyed by FieldSpec.name
    preset_field_values: dict[str, str] = Field(default_factory=dict)


class StartSessionResponse(_Base):
    """Body of ``POST /api/session/start`` response."""

    status: str
    message: str
    session_id: Optional[str] = None
    output_dir: Optional[str] = None


class StopSessionResponse(_Base):
    stopped: bool
    message: str


# ───────────────────────────── monitor + runs ─────────────────────────────


class CandidateRowOut(_Base):
    """One row in the score-history table.

    Mirrors :class:`bracket.ui.monitor.CandidateRow`.
    """

    run_id: str
    role: str
    config_id: str
    score: Optional[float] = None
    final_smoothed: Optional[float] = None
    slope: Optional[float] = None
    n_steps: int = 0
    duration_s: float = 0.0
    disqualified: Optional[str] = None


class LossSeriesOut(_Base):
    """Mirror of :class:`bracket.ui.monitor.LossSeries`.

    Each list is the same length: ``raw[i]`` and ``smoothed[i]`` correspond
    to ``steps[i]``.
    """

    steps: list[int] = Field(default_factory=list)
    raw: list[float] = Field(default_factory=list)
    smoothed: list[float] = Field(default_factory=list)


class MonitorSnapshotOut(_Base):
    """One snapshot consumed by the React Monitor tab.

    Combines :class:`bracket.ui.monitor.MonitorSnapshot` with a few fields
    from the wrapping :class:`bracket.ui.session.SessionState` so the
    frontend can render status without two separate calls.
    """

    # SessionState fields
    session_status: str = "idle"  # idle|running|stopping|done|error
    output_dir: Optional[str] = None
    elapsed_s: float = 0.0
    error_message: Optional[str] = None

    # MonitorSnapshot fields
    status_line: str = ""
    progress_pct: float = 0.0
    completed_runs: int = 0
    total_runs_target: int = 0
    current_run_id: Optional[str] = None
    current_run_steps_done: Optional[int] = None
    current_run_max_steps: Optional[int] = None
    current_loss: Optional[LossSeriesOut] = None
    score_history: list[CandidateRowOut] = Field(default_factory=list)
    setup_status: str = "not started"
    judge_summary: str = ""
    session_done: bool = False
    # Average iterations-per-second for the current run (None until we have
    # at least 2 tfevents samples to compute a delta).
    current_steps_per_sec: Optional[float] = None

    # WS-only metadata; harmless on REST.
    ts: float = 0.0


# ───────────────────────────── gallery ─────────────────────────────


class GalleryItemOut(_Base):
    """One sample image. ``url`` is a relative URL that resolves through
    the static-file mount; ``path`` is the absolute filesystem path (kept
    for parity with the existing Gradio gallery code, but the React client
    should prefer ``url``)."""

    path: str
    url: str
    caption: str
    run_id: str


class GalleryGroupOut(_Base):
    """Mirror of :class:`bracket.ui.monitor.GalleryGroup`."""

    run_id: str
    items: list[GalleryItemOut] = Field(default_factory=list)
    mtime: float = 0.0


# ───────────────────────────── run detail / report ─────────────────────────────


class RunDetailOut(_Base):
    """Per-run detail composed of the ledger row plus derived paths."""

    run_id: str
    role: str
    config_id: str
    score: Optional[float] = None
    n_steps: int = 0
    duration_s: float = 0.0
    disqualified: Optional[str] = None
    config: dict[str, str] = Field(default_factory=dict)
    score_components: dict[str, float] = Field(default_factory=dict)
    judge_report: dict[str, float] = Field(default_factory=dict)
    run_dir: str = ""
    log_path: Optional[str] = None
    sample_dir: Optional[str] = None


class ReportOut(_Base):
    path: str
    content: str


class JudgeStatusOut(_Base):
    configured: bool
    summary: str


class HealthOut(_Base):
    status: str = "ok"
    version: str


# ───────────────────────────── updater ─────────────────────────────


class UpdateStatusOut(_Base):
    """Result of ``GET /api/update/check``."""

    current_version: str
    latest_version: Optional[str] = None
    update_available: bool = False
    release_url: Optional[str] = None
    release_notes: Optional[str] = None
    checked_at: float = 0.0
    error: Optional[str] = None


class TriggerUpdateOut(_Base):
    """Result of ``POST /api/update/apply``."""

    spawned: bool
    message: str
    script: Optional[str] = None
    log_path: Optional[str] = None


# ───────────────────────────── promote / export ─────────────────────────────


class TrainingConfigExportOut(_Base):
    """Result of ``GET /api/runs/{run_id}/training-config``.

    Bundles the candidate's hyperparameters with the session-level
    metadata needed to reproduce or promote the run to a full training:
    the original (full) dataset_toml path, the sample_prompts path, the
    trainer family/type, and a textual ``training_command`` the user can
    paste into a shell to reproduce.
    """

    run_id: str
    role: str
    family: str
    training_type: str
    score: Optional[float] = None
    config: dict[str, str] = Field(default_factory=dict)
    source_dataset_toml: Optional[str] = None
    sample_prompts: Optional[str] = None
    preset_field_values: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class PromoteRunRequest(_Base):
    """Body of ``POST /api/runs/{run_id}/promote``.

    All fields are optional — sensible defaults pick "longer than the
    search but not infinite". Empty / 0 / None ``max_steps`` defaults to
    2000 (~7× a typical search run).
    """

    output_dir: Optional[str] = None       # default: <session>/promoted/<run_id>
    max_steps: int = 2000
    save_every_n_steps: int = 500
    save_state: bool = True
    sample_every_n_steps: Optional[int] = None  # None = every 500 if prompts set
    resume_from: str = ""                  # path to previous LoRA / state dir
    full_dataset_toml: str = ""            # override; default = session source TOML


class PromoteRunResponse(_Base):
    """Result of ``POST /api/runs/{run_id}/promote``."""

    status: str                            # "started" | "conflict" | "bad_request"
    message: str
    promoted_run_id: Optional[str] = None
    output_dir: Optional[str] = None


class ConfigBundleOut(_Base):
    """Result of ``GET /api/config`` — full session config + metadata.

    Designed to round-trip cleanly through the React store: the
    ``request`` field is the exact ``StartSessionRequest`` shape, and
    ``saved_at`` lets the UI surface "imported config from 3 days ago".
    """

    bracket_version: str
    saved_at: float
    name: str = "bracket-session-config"
    request: StartSessionRequest


class ConfigImportIn(_Base):
    """Body of ``POST /api/config/validate`` — round-trips an imported
    config blob through Pydantic for client-side validation help."""

    request: StartSessionRequest
