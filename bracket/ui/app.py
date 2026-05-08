"""Gradio app for Bracket (rebuilt for clarity).

Key UX improvements over v1.0:
  - Cascading **Model** + **Training Type** dropdowns drive which fields
    appear. Required fields end with `*`; optional fields are clearly tagged.
  - Pre-caching (latents + TE outputs for musubi trainers) runs automatically
    as part of the session — no manual scripts.
  - Monitor tab shows: progress bar, live loss plot of the running candidate,
    score-history table, and a samples gallery — not a raw log dump.

Launch:
    python -m bracket.ui.app
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import pandas as pd

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
from bracket.search.optuna_search import OptunaTPESearch
from bracket.ui.monitor import (
    build_snapshot, gallery_groups, gallery_items,
)
from bracket.ui.session import OrchestrationSession


_SESSION = OrchestrationSession()
_LAST_FORM_VALUES: dict[str, str] = {}


# ───────────────────────────── helpers ─────────────────────────────


def _required_label(label: str, required: bool) -> str:
    """The registry already includes ` *` on required labels; here we just
    annotate optional fields explicitly so the UX is unambiguous."""
    if required:
        return label
    if "(optional)" in label.lower():
        return label
    return f"{label} (optional)"


def _max_field_count() -> int:
    """Largest number of trainer fields across all presets — that's how many
    fixed Gradio Textbox slots we allocate (with visibility toggling)."""
    return max(len(p.fields) for p in PRESETS)


_MAX_FIELDS = _max_field_count()


def _preset_fields(family: str, training_type: str) -> tuple[ModelPreset, list[FieldSpec]]:
    preset = get_preset(family, training_type)
    if preset is None:
        # Should never happen with cascading dropdowns; defensive default.
        preset = PRESETS[0]
    return preset, list(preset.fields)


def _on_family_change(family: str) -> tuple:
    """Rebuild Type dropdown choices and reset visible field slots."""
    types = training_types_for(family) or [""]
    new_type = types[0]
    return (
        gr.update(choices=types, value=new_type),
        *_field_updates(family, new_type),
        _preset_notes_md(family, new_type),
    )


def _on_type_change(family: str, training_type: str) -> tuple:
    return (
        *_field_updates(family, training_type),
        _preset_notes_md(family, training_type),
    )


def _field_updates(family: str, training_type: str) -> list:
    """Return a flat list of gr.update(...)s for the N field slots — labels,
    values, and visibility — driven by the selected preset.

    Intentionally does NOT update `info=` because Gradio 6 leaves a stuck
    loading spinner on textboxes whose info wasn't set at construction.
    Help text is shown in the preset_notes panel below the dropdowns instead.
    """
    preset, fields = _preset_fields(family, training_type)
    out: list = []
    for i in range(_MAX_FIELDS):
        if i < len(fields):
            f = fields[i]
            out.append(gr.update(
                label=_required_label(f.label, f.required),
                value=f.default,
                visible=True,
            ))
        else:
            out.append(gr.update(value="", visible=False))
    return out


def _preset_notes_md(family: str, training_type: str) -> str:
    preset, fields = _preset_fields(family, training_type)
    pre_cache = "**Auto pre-cache:** yes (runs once per session)" if preset.needs_pre_cache else "**Auto pre-cache:** not required"
    # Inline field hints since Gradio 6's textbox info= can't be safely updated.
    field_hints_lines: list[str] = []
    for f in fields:
        if f.help:
            req = "**required**" if f.required else "optional"
            field_hints_lines.append(f"- **{f.label.rstrip(' *')}** ({req}): {f.help}")
    field_hints = "\n".join(field_hints_lines) if field_hints_lines else ""
    out = f"### {preset.display_name}\n{preset.notes}\n\n{pre_cache}"
    if field_hints:
        out += "\n\n#### Field guide\n" + field_hints
    return out


def _collect_field_values(family: str, training_type: str, *vals: str) -> dict[str, str]:
    """Map the N positional textbox values to the active preset's field names.
    Anything past the active preset's field count is ignored (those slots are
    hidden in the UI)."""
    preset, fields = _preset_fields(family, training_type)
    out: dict[str, str] = {}
    for i, f in enumerate(fields):
        if i < len(vals):
            out[f.name] = (vals[i] or "").strip()
    return out


# ───────────────────────────── start session ─────────────────────────────


def _stop_session() -> tuple[str, str]:
    if not _SESSION.is_running():
        return ("_Nothing running._", _summary_md())
    _SESSION.stop()
    return ("⏹ Stop requested. The current candidate is being killed; the loop will exit after.", _summary_md())


def _start_session(
    family: str, training_type: str,
    # session-wide
    dataset_toml: str, output_dir: str, sample_prompts: str, resume: str,
    images_per_dataset: int, vram_gb: float,
    budget: int, max_steps: int, wall_secs: int,
    seeds: int, search: str, optuna_startup: int, n_curated: int,
    finals_top_k: int, finals_max_steps: int, finals_seeds: int,
    judge_method: str, judge_base_url: str, judge_model: str,
    judge_loss_weight: float, judge_sample_weight: float,
    # preset-specific (positional, sliced by active preset's fields)
    *preset_field_values: str,
):
    if _SESSION.is_running():
        return "⚠ A session is already running.", _summary_md()

    preset, fields = _preset_fields(family, training_type)
    field_values = _collect_field_values(family, training_type, *preset_field_values)

    # Validate required fields
    missing = [f.label for f in fields if f.required and not field_values.get(f.name)]
    missing += [f.label for f in SESSION_FIELDS if f.required and not (
        dataset_toml.strip() if f.name == "dataset_toml"
        else output_dir.strip() if f.name == "output_dir" else "ok"
    )]
    if missing:
        return ("⚠ Missing required fields:\n- " + "\n- ".join(missing)), _summary_md()

    # Build trainer via the preset factory
    try:
        trainer = preset.trainer_factory(
            **field_values,
            vram_gb=float(vram_gb) if vram_gb else None,
        )
    except Exception as e:  # noqa: BLE001
        return f"⚠ Trainer construction failed: {type(e).__name__}: {e}", _summary_md()

    # Subset
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    subset_dir = out / "subset"
    try:
        subset_toml = build_subset(
            source_toml=Path(dataset_toml).expanduser(),
            target_dir=subset_dir,
            spec=DatasetSubsetSpec(images_per_dataset=int(images_per_dataset), seed=0),
        )
    except Exception as e:  # noqa: BLE001
        return f"⚠ Dataset subset failed: {type(e).__name__}: {e}", _summary_md()

    sp = Path(sample_prompts).expanduser() if sample_prompts.strip() else None
    sample_every = int(max_steps) if sp is not None else None

    judge = None
    if judge_method == "lmstudio" and sp is not None:
        judge = LMStudioJudge(LMStudioJudgeConfig(
            base_url=judge_base_url, model=judge_model,
        ))

    if search == "optuna":
        controller: SearchController = OptunaTPESearch(
            seed=0, n_startup_trials=int(optuna_startup),
        )
    else:
        controller = RandomSearch(seed=0)

    def run_fn() -> OrchestrationResult:
        return orchestrate(
            trainer=trainer, dataset_toml=subset_toml, output_dir=out,
            controller=controller, budget_runs=int(budget),
            max_steps_per_run=int(max_steps),
            max_wall_seconds_per_run=int(wall_secs),
            seeds_per_config=int(seeds),
            n_curated=int(n_curated),
            sample_prompts=sp, sample_every_n_steps=sample_every,
            sample_judge=judge,
            loss_weight=float(judge_loss_weight),
            sample_weight=float(judge_sample_weight),
            stop_event=_SESSION.stop_event,
        )

    finals_fn = None
    if int(finals_top_k) > 0:
        def finals_fn(stage1: OrchestrationResult) -> None:  # noqa: F811
            run_finals_stage(
                trainer=trainer, stage1=stage1, dataset_toml=subset_toml,
                output_dir=out, top_k=int(finals_top_k),
                finals_max_steps=int(finals_max_steps),
                finals_max_wall_seconds=int(int(finals_max_steps) * 12),
                finals_seeds_per_config=int(finals_seeds),
                sample_prompts=sp, sample_every_n_steps=sample_every,
                sample_judge=judge,
                loss_weight=float(judge_loss_weight),
                sample_weight=float(judge_sample_weight),
            )

    total_target = (1 + int(budget)) * int(seeds)
    _SESSION.start(run_fn, finals_fn=finals_fn, output_dir=out, total_runs_target=total_target)
    return f"✓ Started. Output: {out}", _summary_md()


def _summary_md() -> str:
    snap = _SESSION.snapshot()
    if snap.status == "idle":
        return "_No session yet._"
    return (f"**Status:** `{snap.status}` · **runs:** {snap.completed_runs}/{snap.total_runs_target} "
            f"· **elapsed:** {snap.elapsed_s():.0f}s")


# ───────────────────────────── monitor data feeds ─────────────────────────────


_EMPTY_LOSS_DF = pd.DataFrame({"step": [], "loss": [], "series": []})


def _flat_gallery_items(out_dir):
    """Flatten gallery_groups() into a single newest-first list of
    (path, "[run_id]  filename") tuples for one gr.Gallery component.

    Reverted from accordion/per-slot layouts after both `@gr.render` and
    pre-allocated 24-slot designs caused Svelte render-lifecycle hangs on
    tab switch in Gradio 6. One Gallery is rock-solid and keeps the
    newest-first ordering plus visible run grouping via the caption prefix.
    """
    items: list[tuple[str, str]] = []
    for grp in gallery_groups(out_dir):
        for path, name in grp.items:
            items.append((path, f"[{grp.run_id}]  {name}"))
    return items


def _smoothing_to_alpha(smoothing: float) -> float:
    """TensorBoard-style smoothing slider → EMA alpha.

    smoothing=0.0  → alpha=1.0 (no smoothing, line == raw)
    smoothing=0.99 → alpha=0.01 (heavy smoothing)
    Clamped to (0.001, 1.0] so we never divide by zero.
    """
    s = max(0.0, min(0.99, float(smoothing)))
    return max(0.001, 1.0 - s)


def _monitor_refresh(smoothing: float = 0.6):
    """Rebuild every monitor widget from disk + session state.

    `smoothing` is TensorBoard-style (0..0.99); higher = smoother. Translated
    to EMA alpha before feeding into the snapshot so the loss series is
    recomputed live whenever the slider moves.

    NB: gr.LinePlot expects the DataFrame returned DIRECTLY, not wrapped in
    gr.update(...). Wrapping was previously masking the loss values reaching
    the UI (the data was correct in the snapshot, the plot just stayed blank).
    """
    sess_snap = _SESSION.snapshot()
    out_dir = sess_snap.output_dir
    empty_history = pd.DataFrame(columns=[
        "run_id", "role", "config_id", "score", "final_smoothed",
        "slope", "steps", "duration",
    ])
    if out_dir is None:
        return ("_No session started yet._",
                _progress_bar_html(0)["value"],
                _EMPTY_LOSS_DF, empty_history, [])
    alpha = _smoothing_to_alpha(smoothing)
    snap = build_snapshot(
        out_dir, total_runs_target=sess_snap.total_runs_target, ema_alpha=alpha,
    )

    elapsed = sess_snap.elapsed_s()
    status_md = (
        f"**Status:** `{sess_snap.status}` · "
        f"setup: `{snap.setup_status}` · "
        f"completed: **{snap.completed_runs}/{snap.total_runs_target}** "
        f"· elapsed: **{elapsed:.0f}s**"
    )
    if snap.session_done:
        # Session finished — point at the latest run dir (where the loss
        # plot and gallery are sourced from) without claiming it's "running".
        if snap.current_run_id:
            status_md += f"\n\n**Latest run:** `{snap.current_run_id}` (showing its loss curve below)"
    elif snap.current_run_id:
        cs = snap.current_run_steps_done or 0
        ms = snap.current_run_max_steps or 0
        status_md += f"\n\n**Now running:** `{snap.current_run_id}`"
        if ms:
            status_md += f" · step **{cs}/{ms}** ({(cs/ms*100):.1f}%)"
    if snap.judge_summary:
        status_md += f"\n\n{snap.judge_summary}"
    if sess_snap.error_message:
        status_md += f"\n\n```\n{sess_snap.error_message[-1500:]}\n```"

    progress_html_value = _progress_bar_html(snap.progress_pct)["value"]

    # Live loss DataFrame — long form so LinePlot's color="series" splits raw vs smoothed.
    if snap.current_loss is not None and snap.current_loss.steps:
        loss_df = pd.DataFrame({
            "step": snap.current_loss.steps + snap.current_loss.steps,
            "loss": snap.current_loss.raw + snap.current_loss.smoothed,
            "series": (["raw"] * len(snap.current_loss.steps)
                       + ["smoothed"] * len(snap.current_loss.steps)),
        })
    else:
        loss_df = _EMPTY_LOSS_DF.copy()

    rows = []
    for r in snap.score_history:
        rows.append([
            r.run_id, r.role, r.config_id,
            f"{r.score:.6f}" if r.score is not None else (f"DQ: {r.disqualified}" if r.disqualified else "—"),
            f"{r.final_smoothed:.4f}" if r.final_smoothed is not None else "—",
            f"{r.slope:+.4f}" if r.slope is not None else "—",
            r.n_steps,
            f"{r.duration_s:.1f}s",
        ])
    history_df = pd.DataFrame(
        rows,
        columns=["run_id", "role", "config_id", "score", "final_smoothed", "slope", "steps", "duration"],
    )

    return (status_md, progress_html_value, loss_df, history_df,
            _flat_gallery_items(out_dir))


def _progress_bar_html(pct: float) -> dict:
    """Returns a dict shaped like the gr.update result so callers can either
    return its `value` (now standard) or pass the dict to gr.HTML.update()."""
    pct = max(0.0, min(100.0, float(pct)))
    html = (
        "<div style='border:1px solid var(--border-color-primary, #334);"
        "border-radius:6px;background:rgba(120,120,120,0.15);overflow:hidden;height:24px;'>"
        f"<div style='width:{pct:.1f}%;background:#5eead4;height:100%;"
        "transition:width 0.5s;'></div></div>"
        f"<div style='text-align:center;margin-top:4px;font-variant-numeric:tabular-nums;'>{pct:.1f}%</div>"
    )
    return {"value": html}


# ───────────────────────────── results data feeds ─────────────────────────────


def _results_refresh():
    """Returns (report_md, ledger_rows, gallery_items) for the Results tab."""
    sess_snap = _SESSION.snapshot()
    out_dir = sess_snap.output_dir
    if out_dir is None:
        return ("_No session yet._", [], [])
    report = out_dir / "report.md"
    ledger = out_dir / "ledger.jsonl"
    # Regenerate the report on every refresh so it reflects the latest ledger
    # state. Previously we only generated when report.md was absent, which
    # left users staring at a stale snapshot taken after the first few runs.
    if ledger.exists():
        try:
            generate_report(ledger, report)
        except Exception as e:  # noqa: BLE001
            return (f"_Report not yet available: {e}_", [], [])
    md = report.read_text(encoding="utf-8") if report.exists() else "_Report still being generated…_"
    snap = build_snapshot(out_dir, total_runs_target=sess_snap.total_runs_target)
    rows = []
    for r in snap.score_history:
        rows.append([
            r.run_id, r.role, r.config_id,
            f"{r.score:.6f}" if r.score is not None else (f"DQ: {r.disqualified}" if r.disqualified else "—"),
            r.n_steps, f"{r.duration_s:.1f}s",
        ])
    return (md, rows, _flat_gallery_items(out_dir))


# ───────────────────────────── build app ─────────────────────────────


def build_app() -> gr.Blocks:
    families = list_model_families()
    initial_family = families[0]
    initial_types = training_types_for(initial_family)
    initial_type = initial_types[0]
    initial_preset, _ = _preset_fields(initial_family, initial_type)

    # Resolve logo path relative to the repo root so the Gradio app finds it
    # regardless of where it's launched from.
    _logo_path = (Path(__file__).resolve().parent.parent.parent / "assets" / "logo.png").as_posix()
    with gr.Blocks(title="Bracket") as app:
        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=80):
                gr.Image(
                    value=_logo_path, show_label=False, show_download_button=False,
                    container=False, height=64, width=64, interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown(
                    "# Bracket\n"
                    "*Train the same diffusion model eight ways. Pick the one that looks best. "
                    "With a p-value.*"
                )

        with gr.Tab("1 · Setup"):
            with gr.Row():
                family_dd = gr.Dropdown(
                    label="Model *", choices=families, value=initial_family,
                    info="The base diffusion model you're fine-tuning.",
                )
                type_dd = gr.Dropdown(
                    label="Training type *", choices=initial_types, value=initial_type,
                    info="LoRA = small adapter; Full FT = full weight update.",
                )

            preset_notes = gr.Markdown(_preset_notes_md(initial_family, initial_type))

            gr.Markdown(
                "### Model paths\n"
                "_Required fields are marked with `*`. Hover the dropdown above to "
                "see the preset's notes for hints on each field._"
            )
            # Build _MAX_FIELDS textboxes uniformly so gr.update(label/value/visible)
            # works consistently on every slot. Help text lives in the preset notes
            # panel — we don't update `info=` post-construction because Gradio 6
            # gets stuck on a loading spinner when info wasn't set initially.
            field_textboxes: list[gr.Textbox] = []
            for i in range(_MAX_FIELDS):
                if i < len(initial_preset.fields):
                    f = initial_preset.fields[i]
                    tb = gr.Textbox(
                        label=_required_label(f.label, f.required),
                        value=f.default, visible=True,
                    )
                else:
                    tb = gr.Textbox(label="placeholder", value="", visible=False)
                field_textboxes.append(tb)

            gr.Markdown("### Dataset & session")
            dataset_toml = gr.Textbox(
                label="Dataset config TOML *",
                value=SESSION_FIELDS[0].default,
                info=SESSION_FIELDS[0].help,
            )
            output_dir = gr.Textbox(
                label="Session output directory *",
                value=SESSION_FIELDS[1].default,
                info=SESSION_FIELDS[1].help,
            )
            sample_prompts = gr.Textbox(
                label=_required_label(SESSION_FIELDS[2].label, SESSION_FIELDS[2].required),
                value=SESSION_FIELDS[2].default,
                info=SESSION_FIELDS[2].help,
            )
            resume = gr.Textbox(
                label=_required_label(SESSION_FIELDS[3].label, SESSION_FIELDS[3].required),
                value="",
                info=SESSION_FIELDS[3].help,
            )

            with gr.Row():
                images_per_dataset = gr.Slider(
                    label="Images per dataset (subset size)",
                    minimum=4, maximum=64, step=2, value=12,
                    info="Smaller = faster candidates, less accurate ranking.",
                )
                vram_gb = gr.Number(
                    label="VRAM override (GB) — 0 = auto-detect",
                    value=0, precision=1,
                )

            gr.Markdown("### VLM judge (optional)")
            with gr.Row():
                judge_method = gr.Radio(
                    label="Judge", choices=("none", "lmstudio"), value="none",
                    info="Scores generated samples for prompt adherence + visual quality.",
                )
                judge_base_url = gr.Textbox(
                    label="LMStudio base URL", value="http://localhost:1234/v1",
                )
                judge_model = gr.Textbox(
                    label="Model name", value="qwen3-vl-8b-thinking-abliterated",
                )
            with gr.Row():
                judge_loss_weight = gr.Number(label="Loss weight", value=0.3, precision=2)
                judge_sample_weight = gr.Number(label="Sample-quality weight", value=0.7, precision=2)

        with gr.Tab("2 · Run"):
            gr.Markdown("### Search budget")
            with gr.Row():
                budget = gr.Slider(
                    label="Candidate configs (excl. baseline) *",
                    minimum=1, maximum=64, step=1, value=8,
                )
                seeds = gr.Slider(
                    label="Seeds per config",
                    minimum=1, maximum=5, step=1, value=1,
                    info="≥2 enables Welch's t-test confidence intervals in the final report.",
                )
            with gr.Row():
                max_steps = gr.Slider(
                    label="Max steps per run *", minimum=20, maximum=2000, step=10, value=300,
                )
                wall_secs = gr.Slider(
                    label="Wall-time cap per run (s)",
                    minimum=120, maximum=7200, step=60, value=1800,
                )
            with gr.Row():
                search = gr.Radio(
                    label="Search method", choices=("optuna", "random"), value="optuna",
                    info="Optuna TPE learns from history; random is the baseline.",
                )
                optuna_startup = gr.Slider(
                    label="Optuna startup trials (random before TPE kicks in)",
                    minimum=0, maximum=20, step=1, value=5,
                )
            with gr.Row():
                n_curated = gr.Slider(
                    label="Curated warm-start configs (-1 = use all the trainer publishes)",
                    minimum=-1, maximum=10, step=1, value=-1,
                    info="Tries known-good documented configs first, before TPE/random. Each consumes one slot of the budget.",
                )

            gr.Markdown("### Finals stage (optional — verifies stage-1 ranking at higher budget)")
            with gr.Row():
                finals_top_k = gr.Slider(
                    label="Top-K to promote", minimum=0, maximum=8, step=1, value=0,
                )
                finals_max_steps = gr.Slider(
                    label="Finals max steps", minimum=200, maximum=8000, step=100, value=2000,
                )
                finals_seeds = gr.Slider(
                    label="Finals seeds per config", minimum=1, maximum=5, step=1, value=2,
                )

            with gr.Row():
                start_btn = gr.Button("Start orchestration", variant="primary", size="lg")
                stop_btn = gr.Button("Stop orchestration", variant="stop", size="lg")
            start_status = gr.Markdown("_Configure setup, then start._")
            run_summary = gr.Markdown(_summary_md())
            stop_btn.click(fn=_stop_session, inputs=None, outputs=[start_status, run_summary])

        with gr.Tab("3 · Monitor"):
            status_md = gr.Markdown("_No session yet._")
            progress_html = gr.HTML(_progress_bar_html(0)["value"])
            gr.Markdown("### Current candidate · live loss (raw + EMA-smoothed)")
            # TensorBoard-style smoothing slider (0=none, 0.99=heavy).
            # The slider's value is passed into _monitor_refresh and used
            # to recompute the EMA on each refresh.
            smoothing_slider = gr.Slider(
                minimum=0.0, maximum=0.99, value=0.6, step=0.01,
                label="Smoothing (TensorBoard-style EMA — drag to recompute)",
            )
            # Initialise with an empty long-form frame so the component is in
            # the right shape from the start. Refresh callback returns a
            # pandas.DataFrame directly (NOT gr.update — LinePlot in Gradio 6
            # doesn't pick up `value=` through gr.update for plots).
            loss_plot = gr.LinePlot(
                value=_EMPTY_LOSS_DF,
                x="step", y="loss", color="series",
                x_title="step", y_title="loss",
                height=300, show_label=False,
            )
            gr.Markdown("### Score history (one row per completed run)")
            history_table = gr.Dataframe(
                headers=["run_id", "role", "config_id", "score", "final_smoothed", "slope", "steps", "duration"],
                row_count=(0, "dynamic"),
                interactive=False,
            )
            gr.Markdown(
                "### Samples gallery (newest first; `[run_id]` prefix in caption)"
            )
            samples_gallery = gr.Gallery(
                value=[], columns=4, height=420,
                allow_preview=True, show_label=False,
            )

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh now")
                monitor_stop_btn = gr.Button("⏹ Stop orchestration", variant="stop")
            stop_status = gr.Markdown("")
            monitor_stop_btn.click(
                fn=_stop_session, inputs=None, outputs=[stop_status, run_summary],
            )
            _monitor_outputs = [
                status_md, progress_html, loss_plot, history_table, samples_gallery,
            ]
            # Auto-refresh every 5s and on explicit refresh. The smoothing
            # slider is read as an input on each refresh — moving it without
            # clicking refresh just updates the value used on the next tick.
            # We deliberately do NOT bind `smoothing_slider.change` to
            # _monitor_refresh: that introduces a backend round-trip on every
            # pixel of slider drag, which competes with the Timer and was
            # implicated in the tab-switch hangs.
            gr.Timer(5.0).tick(
                fn=_monitor_refresh, inputs=[smoothing_slider],
                outputs=_monitor_outputs,
            )
            refresh_btn.click(
                fn=_monitor_refresh, inputs=[smoothing_slider],
                outputs=_monitor_outputs,
            )

        with gr.Tab("4 · Results"):
            report_md = gr.Markdown("_Will populate when session finishes._")
            ledger_df = gr.Dataframe(
                headers=["run_id", "role", "config_id", "score", "steps", "duration"],
                row_count=(0, "dynamic"),
                interactive=False,
            )
            gr.Markdown(
                "### Samples (newest first; `[run_id]` prefix in caption)"
            )
            results_gallery = gr.Gallery(
                value=[], columns=4, height=520,
                allow_preview=True, show_label=False,
            )

            results_refresh_btn = gr.Button("🔄 Refresh results")
            _results_outputs = [report_md, ledger_df, results_gallery]
            results_refresh_btn.click(
                fn=_results_refresh, inputs=None,
                outputs=_results_outputs,
            )

        # Wiring: dropdown changes → field updates
        family_dd.change(
            fn=_on_family_change, inputs=[family_dd],
            outputs=[type_dd, *field_textboxes, preset_notes],
        )
        type_dd.change(
            fn=_on_type_change, inputs=[family_dd, type_dd],
            outputs=[*field_textboxes, preset_notes],
        )

        # Wiring: start button
        start_btn.click(
            fn=_start_session,
            inputs=[
                family_dd, type_dd,
                dataset_toml, output_dir, sample_prompts, resume,
                images_per_dataset, vram_gb,
                budget, max_steps, wall_secs,
                seeds, search, optuna_startup, n_curated,
                finals_top_k, finals_max_steps, finals_seeds,
                judge_method, judge_base_url, judge_model,
                judge_loss_weight, judge_sample_weight,
                *field_textboxes,
            ],
            outputs=[start_status, run_summary],
        )

        gr.Markdown(
            "---\n"
            "_Bracket v1.1 — auto pre-cache · cascading model picker · "
            "TPE search · multi-seed · LMStudio judge · 2-stage finals_"
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(prog="bracket-ui")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_app()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
