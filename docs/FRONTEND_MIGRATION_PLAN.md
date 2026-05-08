# OmniSteer-Diffusion · Frontend Migration Plan

Status: Proposal · Author: frontend architecture pass · Target: replace Gradio (`omnisteer_diffusion/ui/app.py`) with a React + Vite SPA backed by a FastAPI surface that already coexists in the package's dep list.

---

## 1. Executive summary

Gradio got us to v1.1, but the four-tab app is now hitting Gradio's ceilings. The Monitor tab polls every 5 s through a re-rendered Svelte tree and has had two documented stability incidents (`@gr.render` lifecycle hangs on tab switch, `info=` post-construction loading-spinner bug, `gr.update`-vs-DataFrame plot blanking). Smoothing-slider drag, comparison views, deep links to a single run, keyboard shortcuts, and a real samples-grid lightbox are physically impossible inside Gradio without iframe gymnastics. Mobile is unusable.

Migrating buys: real WebSocket-driven loss streaming, Linear/Vercel-grade polish, branded shell, accessible focus management, deep-linkable runs (`?run=cand-007`), client-side smoothing recompute, drag-and-drop TOML upload, comparison mode for samples, dark/light theming, keyboard shortcuts, and a printable Results page. It costs: an explicit FastAPI surface (already a transitive dep — uvicorn ships with Gradio), a Vite build pipeline, and a one-time deploy refactor to single-port static-mount. Net positive — the Gradio quirks documented in `ui/app.py` (10 separate `# NB:` comments) will not survive contact with shadcn/ui + Recharts + TanStack Query.

**Recommended stack (one line):** Vite + React 19 + TypeScript strict + Tailwind v4 + shadcn/ui + Recharts + TanStack Query + Zustand + TanStack Router + native WebSocket + React Hook Form + Zod + Sonner + Motion + Lucide + Biome + lefthook, served from a single FastAPI process in production via `StaticFiles`.

---

## 2. Recommended stack with versions

| Concern | Pick | Version |
|---|---|---|
| Build | Vite | ^7.0 |
| UI runtime | React + ReactDOM | ^19.1 |
| Language | TypeScript (strict, `noUncheckedIndexedAccess`) | ^5.7 |
| Styling | Tailwind CSS | ^4.0 |
| Components | shadcn/ui (Radix primitives + Tailwind) | latest CLI |
| Icons | Lucide React | ^0.468 |
| Charts | Recharts | ^2.15 |
| Server state | TanStack Query | ^5.62 |
| UI state | Zustand | ^5.0 |
| Router | TanStack Router (file-based) | ^1.95 |
| Forms | React Hook Form + Zod | ^7.54 / ^3.24 |
| Animations | Motion (Framer Motion successor) | ^11.15 |
| Toasts | Sonner | ^1.7 |
| Code/diff blocks | Shiki + shiki-magic-move | ^1.24 |
| WebSocket | Native `WebSocket` + small reconnect wrapper (`partysocket` if hosted later) | — / ^1.0 |
| Lint+format | Biome | ^1.9 |
| Pre-commit | lefthook | ^1.10 |

**shadcn/ui over MUI / Chakra / Mantine / Ant.** shadcn ships unstyled Radix primitives wrapped in Tailwind classes that are *copied into your repo*, not imported from a versioned package. That matters here because the Linear/Vercel/Resend aesthetic the brief asks for (zinc neutrals, generous whitespace, one accent — keep the existing `#5eead4` teal) is achievable only when you own the styles. MUI/Chakra/Ant come with strong opinions you'd fight on every component; Mantine is the closest second but has less momentum and a heavier runtime. shadcn = aesthetic ceiling + zero lock-in.

**Recharts over Visx / uPlot / Tremor.** Up to 5000 points, refreshed every 5 s. uPlot is 5–10× faster but the API is imperative and the React wrapper (`uplot-react`) is thin; Visx requires hand-assembling axes, scales, and tooltips for every chart; Tremor is opinionated dashboards (good Linear-aesthetic, but locks the chart axis). Recharts is declarative, has built-in tooltip/legend/brush, animates smoothly under 1k–5k points, and we'll virtualise at 5k+ via a downsampling step in the WS handler (LTTB on the server, send max 1000 points). If we ever need 60-fps streaming over 10k points, swap to uPlot in one component without touching the rest.

**TanStack Query + Zustand over Redux / Context.** Server state is 90% of this app — runs, ledger, snapshots, judge status. TanStack Query handles caching, retries, staleTime, optimistic updates, and the WS bridge (`queryClient.setQueryData` from a WS message). Zustand owns the 10% of UI state that isn't server-derived: smoothing slider value, comparison-mode selection, theme. Redux is overkill; Context re-renders the world.

**React Hook Form + Zod.** The Setup tab has cascading required/optional fields driven by the registry (`PRESETS`, `SESSION_FIELDS`). RHF + Zod's `discriminatedUnion` map cleanly onto `ModelPreset.fields`, with one Zod schema per preset, switched by `model_family` + `training_type`. Validation errors render inline; the form's `isValid` gates the Start button.

**TanStack Router over React Router 7.** Type-safe params (`/runs/$runId` is typed end-to-end), file-based routes, built-in search-param schemas. Deep-link support for `?run=cand-007` is one Zod schema, not a useEffect dance.

**Motion for animations.** Tab transitions, gallery item entry, skeletons → content fade. Motion is the same engine as Framer Motion, smaller bundle, drop-in. Use sparingly — every animation under 200 ms.

**Native WebSocket + small reconnect wrapper.** No reason to pull in socket.io for a single-channel server-push stream. ~30 LOC wrapper handles exponential backoff, page-visibility pause, and resync via `GET /sessions/current/snapshot` after reconnect.

**Sonner for toasts.** Default shadcn pick. Sub-1 KB, accessible, stacks cleanly.

**Shiki + shiki-magic-move.** The Results tab will offer a "config diff" between two runs. shiki-magic-move animates token-level diffs of TOML/JSON. Treat as Phase-2 polish — ship without it, add when the diff view lands.

**Biome over ESLint + Prettier.** One binary, no peer-dep matrix, 10× faster, sufficient rule set for this size of app. lefthook over husky because it's a single Go binary, no `node_modules` shenanigans on the pre-commit path.

---

## 3. Backend API design

The orchestrator is already FastAPI-ready (`fastapi>=0.110`, `uvicorn[standard]>=0.27` in `pyproject.toml`). We keep `OrchestrationSession` as the in-process state singleton and expose it through a FastAPI router. v0.1 = single-user local tool, no auth. CORS open to `localhost:5173` in dev, same-origin in prod.

### REST surface

| Method | Path | Body | Response | Notes |
|---|---|---|---|---|
| GET | `/api/presets` | — | `Preset[]` | Drives cascading dropdown. `families`, `training_types`, `fields`, `notes`, `needs_pre_cache`. |
| GET | `/api/presets/{family}/{type}` | — | `Preset` | Resolves one preset for prefill. |
| GET | `/api/session-fields` | — | `FieldSpec[]` | The 4 always-shown session fields. |
| POST | `/api/sessions/validate` | `SessionConfig` | `{ ok: bool, missing: string[], errors: Record<string,string> }` | Server-side mirror of the Zod schema, single source of truth. |
| POST | `/api/sessions` | `SessionConfig` | `{ session_id: string, output_dir: string }` | Starts orchestration in the existing background thread. 409 if already running. |
| DELETE | `/api/sessions/current` | — | `{ stopped: bool }` | Wraps `OrchestrationSession.stop()` — kills the active subprocess, sets stop_event. |
| GET | `/api/sessions/current` | — | `MonitorSnapshot \| null` | One-shot snapshot. Used on initial mount and on WS reconnect. |
| GET | `/api/sessions/current/runs` | — | `RunRow[]` | Score history. |
| GET | `/api/sessions/current/runs/{run_id}` | — | `RunDetail` | Includes config, full loss series, sample list, judge report. |
| GET | `/api/sessions/current/runs/{run_id}/loss?since={step}` | — | `LossSeries` | Raw points only — smoothing is client-side. `since` enables incremental fetch. |
| GET | `/api/sessions/current/runs/{run_id}/gallery` | — | `GalleryItem[]` | Image URLs (proxied — see file-serving below). |
| GET | `/api/sessions/current/report` | — | `{ markdown: string, generated_at: string }` | Regenerated on every call (cheap; mirrors current `_results_refresh` behaviour). |
| GET | `/api/judge/status` | — | `{ configured: bool, base_url: string, model: string, reachable: bool }` | Pings LMStudio `/v1/models`. |
| GET | `/api/curated/{family}/{type}` | — | `CuratedConfig[]` | Optional: surface what `n_curated` will warm-start with, so the user sees them. |
| GET | `/api/health` | — | `{ ok: true, version: string }` | Liveness. |

### WebSocket

| Path | Direction | Message | Cadence |
|---|---|---|---|
| `/ws/session` | Server → Client | `{ type: "snapshot", data: MonitorSnapshot }` | Every 2 s while a session is in-flight. Idle (no traffic) when no session. |
| | Server → Client | `{ type: "log", data: { line: string, level: string, ts: number } }` | Per log line, batched at ≤30 Hz. |
| | Server → Client | `{ type: "run_completed", data: RunRow }` | Push on every ledger append. Triggers TanStack Query invalidate. |
| | Server → Client | `{ type: "session_state", data: { status: "idle"\|"running"\|"stopping"\|"done"\|"error" } }` | On state transition. |
| | Client → Server | `{ type: "ping" }` | Keepalive every 30 s, server responds `pong`. |

This replaces the `gr.Timer(5.0)` polling. The 2-s server cadence is conservative; bumping to 1 s is trivial. Reconnect strategy: client backs off 1 s → 2 s → 4 s → 8 s (max), and on first message after reconnect issues `GET /api/sessions/current` to resync.

### File serving (sample images)

The current code passes absolute Windows paths (`I:\AI\OmniSteer-Diffusion\runs\ui-002\runs\cand-007-s0-...\output\sample\foo.png`) directly to `gr.Gallery`, which works because Gradio handles the file-resolution. The React frontend can't fetch `file://` URLs.

Solution: a single virtual mount.

```
GET /files/runs/{run_id}/{path:path}
```

Resolves to `{session.output_dir}/runs/{run_id}/{path}`, with hard guards:
1. Reject any path containing `..` or absolute markers.
2. Resolve and verify the result is `Path(session.output_dir).resolve() / "runs"` is a parent.
3. Whitelist content types: `.png`, `.jpg`, `.jpeg`, `.webp`, `.txt`, `.json`.
4. Add `Cache-Control: public, max-age=31536000, immutable` — sample filenames already encode `<step>_<idx>_<seed>` so they're effectively immutable.

Alternative considered: signed URLs. Overkill for v0.1 single-user local. Document the same path-traversal-guard approach as the auth seam for a future hosted version.

### CORS (dev only)

```py
allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"]
allow_credentials=True
allow_methods=["*"]; allow_headers=["*"]
```

In prod the React bundle is served from FastAPI itself, so CORS is moot.

### Router signatures (pseudo-code)

```py
# omnisteer_diffusion/api/router.py
router = APIRouter(prefix="/api")

@router.get("/presets") -> list[PresetDTO]
@router.post("/sessions", status_code=201) -> SessionStarted
    # body: SessionConfigDTO; raises 409 if SESSION.is_running()
@router.delete("/sessions/current") -> StopResult
@router.get("/sessions/current") -> Optional[MonitorSnapshotDTO]
@router.get("/sessions/current/runs") -> list[RunRowDTO]
@router.get("/sessions/current/runs/{run_id}") -> RunDetailDTO
@router.get("/sessions/current/runs/{run_id}/loss") -> LossSeriesDTO
@router.get("/sessions/current/report") -> ReportDTO

# omnisteer_diffusion/api/ws.py
@app.websocket("/ws/session")
async def ws_session(socket: WebSocket): ...
```

DTOs are pydantic models that mirror the dataclasses in `ui/monitor.py` exactly — no shape changes needed in the orchestrator.

---

## 4. Gradio → React component mapping

Exhaustive, by tab. "UX upgrade" column lists wins available because shadcn/ui isn't constrained the way Gradio is.

### Setup tab

| Gradio | Behaviour | React replacement | Import | UX upgrade | Tree location |
|---|---|---|---|---|---|
| `gr.Dropdown` (Model) | Cascading — drives Type + fields | `Combobox` | `@/components/ui/combobox` | Fuzzy search via `cmdk`; keyboard-first; preset preview on hover | `routes/setup/PresetPicker.tsx` |
| `gr.Dropdown` (Training type) | Reset on family change | `Select` | `@/components/ui/select` | Disabled options shown for not-yet-supported combos | `PresetPicker.tsx` |
| `gr.Markdown` (preset notes) | Rebuilt on dropdown change | `<Card>` + `react-markdown` | `@/components/ui/card`, `react-markdown` | Field-guide bullets render with icons; collapsible | `PresetPicker.tsx` |
| `gr.Textbox` ×N (preset fields) | Hidden via `visible=False` for inactive presets | `Input` + `Label` + `FormMessage` | `@/components/ui/{input,form}` | Per-field icon (folder/file); browse button via Tauri-or-clipboard fallback; required asterisk styled | `routes/setup/PresetFields.tsx` |
| `gr.Textbox` (Dataset TOML) | Plain textbox | `Input` + drag-and-drop dropzone | `@/components/ui/input` + `react-dropzone` | Drop a `.toml` to autofill path AND show parse preview (resolution buckets, image count) | `routes/setup/DatasetSection.tsx` |
| `gr.Textbox` (Output dir) | Plain | `Input` | `@/components/ui/input` | Inline "create if missing" toggle | `DatasetSection.tsx` |
| `gr.Textbox` (Sample prompts) | Path | `Input` + popover preview | `@/components/ui/popover` | Hover popover shows the first 3 prompts | `DatasetSection.tsx` |
| `gr.Textbox` (Resume from) | Optional path | `Input` | `@/components/ui/input` | — | `DatasetSection.tsx` |
| `gr.Slider` (Images per dataset) | 4–64 step 2 | `Slider` w/ marks + numeric input | `@/components/ui/slider` | Synced numeric input; tick marks at 4/12/24/64 with labels | `routes/setup/SubsetSection.tsx` |
| `gr.Number` (VRAM override) | 0 = auto | `Input type=number` + auto badge | `@/components/ui/input` | Detected-VRAM hint pulled from `/api/health` | `SubsetSection.tsx` |
| `gr.Radio` (Judge method) | none / lmstudio | `RadioGroup` | `@/components/ui/radio-group` | Live "reachable" badge polled from `/api/judge/status` | `routes/setup/JudgeSection.tsx` |
| `gr.Textbox` (LMStudio base URL) | Conditional | `Input` (disabled when `none`) | — | Shows "↻ Test connection" button | `JudgeSection.tsx` |
| `gr.Textbox` (Model name) | Conditional | `Combobox` populated from `/v1/models` | — | Auto-discovers loaded models | `JudgeSection.tsx` |
| `gr.Number` (Loss weight / Sample weight) | 2 numbers, no constraint | Linked dual-handle slider, sums to 1.0 | `@/components/ui/slider` (custom) | Visual ratio bar; click to flip 0.3/0.7 ↔ 0.7/0.3 | `JudgeSection.tsx` |

### Run tab

| Gradio | Behaviour | React replacement | Import | UX upgrade | Tree location |
|---|---|---|---|---|---|
| `gr.Slider` (Budget) | 1–64 | `Slider` + numeric | `@/components/ui/slider` | Estimated wall-clock badge: `≈ 8 × 1800s = 4h` | `routes/run/BudgetSection.tsx` |
| `gr.Slider` (Seeds per config) | 1–5 | `Slider` | — | "Enables Welch's t-test" badge appears at ≥2 | `BudgetSection.tsx` |
| `gr.Slider` (Max steps / Wall time) | Two sliders | `Slider` ×2 | — | Dual-axis card with combined ETA | `BudgetSection.tsx` |
| `gr.Radio` (Search method) | optuna / random | `Tabs` (segmented control) | `@/components/ui/tabs` | Visual: "TPE learns" with mini sparkline | `routes/run/SearchSection.tsx` |
| `gr.Slider` (Optuna startup) | 0–20 | `Slider` (disabled when search=random) | — | — | `SearchSection.tsx` |
| `gr.Slider` (n_curated) | -1 = all | `Slider` + "Show curated configs" link | — | Side panel lists what curated configs the trainer will warm-start with (calls `/api/curated/...`) | `SearchSection.tsx` |
| `gr.Slider` ×3 (Finals) | top-K, max_steps, seeds | Collapsible `Card` with 3 sliders | `@/components/ui/collapsible` | Default-collapsed; "Skip finals" toggle = top_k=0 | `routes/run/FinalsSection.tsx` |
| `gr.Button` (Start / Stop) | Side by side | `Button` (primary) + `Button` (destructive) | `@/components/ui/button` | Sticky in a bottom bar; keyboard `s` to start, `Esc` to stop | `components/run/RunControls.tsx` |
| `gr.Markdown` (start_status / run_summary) | Two markdowns | `Toaster` + `StatusBadge` | `sonner`, `@/components/run/StatusBadge.tsx` | Toast on start/stop instead of inline text spam | `RunControls.tsx` |

### Monitor tab

| Gradio | Behaviour | React replacement | Import | UX upgrade | Tree location |
|---|---|---|---|---|---|
| `gr.Markdown` (status) | Multi-line markdown rebuilt every 5 s | `<SessionHeader>` w/ live badges | `@/components/monitor/SessionHeader.tsx` | Status pill color-coded, judge summary as separate badge with hover details | `routes/monitor/index.tsx` |
| `gr.HTML` (progress bar) | Inline-styled HTML | `<Progress>` from shadcn | `@/components/ui/progress` | Striped animation while running, smooth `transition-all 500ms` | `SessionHeader.tsx` |
| `gr.Slider` (smoothing) | EMA recompute via 5 s timer roundtrip | `<Slider>` w/ client-side EMA | `@/components/ui/slider` | Recomputes locally — drag is silky | `components/monitor/LossChart.tsx` |
| `gr.LinePlot` (loss raw + smoothed) | Re-rendered each tick | `<LineChart>` from Recharts, two `<Line>` series | `recharts` | Tooltip with both values, brush for x-axis zoom, hover crosshair | `components/monitor/LossChart.tsx` |
| `gr.Dataframe` (score history) | 8 columns, dynamic rows | `<DataTable>` (shadcn + TanStack Table) | `@/components/ui/data-table` | Sortable columns, sparkline column ("score by run"), row click → run detail drawer | `components/monitor/RunHistoryTable.tsx` |
| `gr.Gallery` (samples) | Single flat gallery, captions like `[run]  file` | `<GalleryAccordion>` (one accordion per run, lazy-load images) | `@/components/monitor/GalleryAccordion.tsx` | Native `<img loading="lazy">`, lightbox on click, comparison-mode select-2 | `routes/monitor/index.tsx` |
| `gr.Button` (Refresh) | Manual refresh | Removed (WS push). Keyboard `r` forces `queryClient.invalidateQueries(...)` | — | — | — |
| `gr.Button` (Stop) | Duplicated on tab | Sticky bottom bar (shared with Run tab) | `components/run/RunControls.tsx` | Always visible regardless of route | persistent layout |

### Results tab

| Gradio | Behaviour | React replacement | Import | UX upgrade | Tree location |
|---|---|---|---|---|---|
| `gr.Markdown` (report) | Regenerated on every refresh | `<Markdown>` via `react-markdown` + `remark-gfm` | `react-markdown` | Sticky TOC, copy-as-markdown button, print stylesheet | `routes/results/Report.tsx` |
| `gr.Dataframe` (ledger) | 6 cols | `<DataTable>` | `@/components/ui/data-table` | Filter by role; "compare 2-3" multi-select | `components/results/LedgerTable.tsx` |
| `gr.Gallery` (samples) | Flat list | `<ResultsGallery>` w/ comparison mode | `@/components/results/ResultsGallery.tsx` | Pick 2-3 runs, show sample[i] side-by-side at the same prompt index | `routes/results/index.tsx` |
| `gr.Button` (Refresh) | Manual | Removed (TanStack Query auto-refetch + WS) | — | — | — |

---

## 5. Information architecture & nav

Today: 4 numbered tabs in a row, no persistent state indicator.

**Pick:** persistent left rail (collapsible to icons) + global command palette (`Ctrl/Cmd+K`).

```
╭───────────────────────────────────────────────────────────╮
│ ╔═════════╗                                               │
│ ║ OmniS.  ║   Setup ▸ select preset                       │
│ ╠═════════╣                                               │
│ ║ Setup   ║   <main panel>                                │
│ ║ Run     ║                                               │
│ ║ Monitor ║                                               │
│ ║   ●live ║                                               │
│ ║ Results ║                                               │
│ ╠═════════╣                                               │
│ ║ status  ║   <persistent run-control bar at bottom>      │
│ ║ pill    ║                                               │
│ ╚═════════╝                                               │
╰───────────────────────────────────────────────────────────╯
```

The Monitor entry shows a green pulse dot when a session is running (driven by WS `session_state`). The status pill at the bottom of the rail mirrors the snapshot status (`idle / running 4/8 / stopping / done / error`) and is clickable to open the Monitor route.

Routes (TanStack Router file-based):

```
src/routes/
├── __root.tsx           # AppShell: sidebar + run-controls bar + Toaster
├── index.tsx            # → redirect to /setup
├── setup.tsx
├── run.tsx
├── monitor.tsx
├── monitor.$runId.tsx   # drawer route — overlays Monitor
├── results.tsx
└── results.$runId.tsx   # ?compare=cand-002,cand-005 supported via search schema
```

Command palette (Ctrl/Cmd+K) lives in `__root.tsx`, indexes routes + presets + recent runs, jumps to `/monitor/$runId` with one keystroke.

---

## 6. Page-level wireframes

### Setup

```
┌─ AppShell ────────────────────────────────────────────────────────┐
│  Setup                                                             │
│  ┌─ Preset ──────────────────────────────┐  ┌─ Notes ───────────┐│
│  │ Model:        [SDXL          ▾]        │  │ ## SDXL · LoRA    ││
│  │ Training type:[LoRA          ▾]        │  │ Wraps sd-scripts/ ││
│  └────────────────────────────────────────┘  │ sdxl_train_…      ││
│                                              │ Auto pre-cache: no││
│  ┌─ Model paths ─────────────────────────┐  │ Field guide:      ││
│  │ SDXL base path *      [I:/AI/…    ⊞] │  │  • SDXL base (req)││
│  │ sd-scripts dir *      [I:/AI/…    📁]│  │    HF snapshot or…││
│  │ Trainer venv python * [I:/AI/…    ⊞] │  └───────────────────┘│
│  └────────────────────────────────────────┘                       │
│                                                                   │
│  ┌─ Dataset & session ────────────── drop a .toml here ─────────┐│
│  │ Dataset TOML *  [⤓ I:/AI/…toml]   parsed: 124 imgs, 4 buckets││
│  │ Output dir *    [I:/AI/…       ]  ☑ create if missing        ││
│  │ Sample prompts  [I:/AI/…       ]  ⓘ preview                  ││
│  │ Resume from     [               ]                            ││
│  │ Subset size     [○──────●───────○]  12   VRAM  [auto · 32GB] ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─ VLM judge (optional) ───────────────────────────────────────┐│
│  │ Method  ( ) none  (●) lmstudio   ●reachable                  ││
│  │ Base URL [http://localhost:1234/v1] ↻test                    ││
│  │ Model    [qwen3-vl-8b-thinking-abliterated         ▾]        ││
│  │ Weights  loss [────●──────] 0.3    sample [────────●─] 0.7   ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                   │
│  [persistent footer: Idle ·                       Start ▸]       │
└───────────────────────────────────────────────────────────────────┘
```

Empty state (no preset selected): centered card with "Pick a model to begin" + the family chooser inline.
Loading: skeleton on the field rows during preset switch (200 ms fade).
Error: inline `FormMessage` per field; Sonner toast for backend validation failures from `POST /api/sessions/validate`.

### Run

```
┌─ Search budget ─────────────────────────────────────────────────┐
│ Candidate configs *    [○──●─────────] 8        ≈ 4h wall      │
│ Seeds per config       [●─○]            1        +CI at ≥2     │
│ Max steps per run *    [───●─────] 300                         │
│ Wall-time cap (s)      [────●────] 1800                        │
│ Search method   [ Optuna │ Random ]    Startup [───●───] 5     │
│ Curated warm-start     [───────●] -1   ⓘ show curated configs  │
└─────────────────────────────────────────────────────────────────┘

┌─ Finals stage ▾ ─────────────────────────────────────────────── ┐
│ ☑ Run finals stage                                              │
│ Top-K [○──●──○] 3   Max steps [────●──] 2000   Seeds [●──○] 2   │
└─────────────────────────────────────────────────────────────────┘

[footer: Idle · Output → I:/AI/…/runs/ui-002 ·         Start ▸]
```

Empty: enabled, defaults populated.
Loading: Start button shows spinner + "Queueing setup…" optimistic state for ~300 ms.
Error: toast with the `OrchestrationSession` exception message (truncated at 1500 chars, full in a modal).

### Monitor

```
┌─ SessionHeader ─────────────────────────────────────────────────┐
│ ●running  cand-005  step 142/300 (47.3%)    elapsed 6m 21s     │
│ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░  3/8 runs                  │
│ Judge: ✓ LMStudio · 2/3 scored · 18 imgs · mean 7.4/10         │
└─────────────────────────────────────────────────────────────────┘

┌─ Live loss · cand-005 ──────────────────────────────────────────┐
│   ↑                                                             │
│ 0.45                                                            │
│   ┊ ╲╱╲   raw                                                   │
│   ┊  ╲___ smoothed                                              │
│ 0.20  ╲___                                                      │
│   └────┴───────────────→ step                                  │
│ Smoothing  [─────●──────────]  0.6   (client-side EMA)          │
└─────────────────────────────────────────────────────────────────┘

┌─ Score history ────────────────────────────────────────────── ─┐
│ run_id        role       score  final  slope  steps  duration  │
│ cand-002-s0   candidate  0.241  0.18   -0.01  300    98.2s     │
│ cand-001-s0   baseline   0.288  0.23   -0.00  300   102.1s ⌃   │
│ ...                                                          ▾ │
└─────────────────────────────────────────────────────────────────┘

┌─ Samples gallery ───────────────────────────────────────────────┐
│ ▾ cand-005 (live)            ▾ cand-002             ▾ cand-001 │
│ [thumb][thumb][thumb][thumb] [thumb][thumb][thumb]  [thumb]…   │
└─────────────────────────────────────────────────────────────────┘
```

Empty (no session): centered "No session yet" + a `Start a session →` button that routes to `/setup`.
Loading: skeleton on each card; loss chart shows a shimmering 1-line placeholder.
Error: status pill flips red, error message in a collapsible card under the header (full traceback in modal).

### Results

```
┌─ Report (markdown) ─────────────────┐  ┌─ TOC ─────────┐
│ # OmniSteer-Diffusion Run Report     │  │ • Summary     │
│                                      │  │ • Best config │
│ ## Best config                       │  │ • Ledger      │
│ id: 7a3f… · score: 0.241 ± 0.008    │  │ • Samples     │
│ ...                                  │  └───────────────┘
└──────────────────────────────────────┘  [Print  Copy MD]

┌─ Ledger ───────────────────────────────────────────────────────┐
│ ☐ ☐ ☐  filter [all ▾]                                          │
│ ☑ cand-005  candidate  0.241  300  102s        ▶ details       │
│ ☑ cand-002  candidate  0.252  300   98s        ▶ details       │
│ ☐ cand-001  baseline   0.288  300  102s        ▶ details       │
│ Compare selected (2)                                            │
└─────────────────────────────────────────────────────────────────┘

┌─ Comparison · prompt 0 ─────────────────────────────────────────┐
│   cand-005           |   cand-002                               │
│   [image]            |   [image]                                │
│   prompt_adh: 8.0    |   prompt_adh: 7.5                        │
└─────────────────────────────────────────────────────────────────┘
```

Empty: "Session has not finished yet. Live progress in Monitor →".
Loading: report skeleton; gallery thumbs as 16:9 grey blocks with shimmer.
Error: report card shows "Report failed: {e}" with a "Regenerate" button calling `GET /api/sessions/current/report`.

---

## 7. UX upgrades worth doing

Eleven wins, in priority order.

1. **WebSocket-driven loss streaming** — Server pushes raw loss frames at trainer cadence; chart updates in <50 ms, not 5 s. Implementation: WS message `{ type: "loss_frame", run_id, step, raw }` appended to a Zustand-backed ring buffer per run; LineChart reads from the buffer.

2. **Client-side smoothing slider** — EMA recomputed in JS from the raw buffer, no backend roundtrip. Drag at 60 fps. Implementation: pure function `applyEMA(raw[], alpha) → smoothed[]` memoised via `useMemo`.

3. **Drag-and-drop dataset TOML** — Drop a `.toml`, browser parses it (we add a tiny TOML parser like `smol-toml`, ~3 KB), shows resolution buckets + image count before submission. Implementation: `react-dropzone` + client-side parse; path is auto-filled if running locally with Tauri.

4. **Sticky run controls** — Start/Stop in a persistent bottom bar so Monitor isn't the only place to stop. Already in the wireframes. Implementation: `__root.tsx` slot.

5. **Comparison mode in Results gallery** — Multi-select 2–3 runs in the ledger table, get a synced grid that shows sample[i] from each side-by-side. Massive upgrade vs Gradio's flat-list. Implementation: Zustand `comparisonStore`, `ResultsGallery` component, prompt index slider.

6. **Deep-linkable runs** — `/results?run=cand-007` opens the run-detail drawer over the Results page. URL-as-state via TanStack Router search schema, so refresh + share + back-button all work. Implementation: `useSearch({ from: '/results' })`.

7. **Sparkline column in score history** — Per-row mini loss curve in the table. Implementation: tiny inline `<LineChart width={80} height={24}>` per row, fed by the cached series from TanStack Query.

8. **Keyboard shortcuts** — `s` start, `Esc` stop, `r` force refresh, `Cmd+K` palette, `1-4` jump tabs, `[` / `]` cycle smoothing, `c` toggle comparison mode. Implementation: `react-hotkeys-hook` (~1 KB).

9. **Optimistic Start** — Click Start, button instantly shows "Queueing setup…" with a spinner; server response confirms or reverts. Implementation: TanStack Query `useMutation` with `onMutate` setting status to `running` in the snapshot cache.

10. **Dark / light theming** — Tailwind v4's `@theme` blocks, default = dark (matches the existing `--bg: #0b0d12`), toggle in the rail. Implementation: shadcn's `ThemeProvider` pattern.

11. **Print-friendly Results** — `@media print { .no-print { display: none } }`, gallery becomes a 4-up grid, ledger paginates cleanly. One stylesheet, no JS.

12. **Accessible focus management** — Every Radix primitive ships with focus trap + ARIA. The Gradio version has none. Implementation: free with shadcn.

---

## 8. File / folder structure

Replace the existing `frontend/index.html` static file. The new root is still `i:\AI\OmniSteer-Diffusion\frontend\` (re-using the directory keeps the FastAPI mount path identical in prod).

```
frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── biome.json
├── lefthook.yml
├── tailwind.config.ts
├── index.html
├── public/
│   └── favicon.svg
└── src/
    ├── main.tsx
    ├── routeTree.gen.ts            # TanStack Router generated
    ├── routes/
    │   ├── __root.tsx
    │   ├── index.tsx               # redirect → /setup
    │   ├── setup.tsx
    │   ├── run.tsx
    │   ├── monitor.tsx
    │   ├── monitor.$runId.tsx
    │   ├── results.tsx
    │   └── results.$runId.tsx
    ├── components/
    │   ├── ui/                     # shadcn primitives (copied in)
    │   │   ├── button.tsx
    │   │   ├── card.tsx
    │   │   ├── combobox.tsx
    │   │   ├── data-table.tsx
    │   │   ├── form.tsx
    │   │   ├── input.tsx
    │   │   ├── progress.tsx
    │   │   ├── select.tsx
    │   │   ├── slider.tsx
    │   │   ├── tabs.tsx
    │   │   └── …
    │   ├── layout/
    │   │   ├── AppShell.tsx
    │   │   ├── Sidebar.tsx
    │   │   └── RunControlsBar.tsx
    │   ├── setup/
    │   │   ├── PresetPicker.tsx
    │   │   ├── PresetFields.tsx
    │   │   ├── DatasetSection.tsx
    │   │   ├── SubsetSection.tsx
    │   │   └── JudgeSection.tsx
    │   ├── run/
    │   │   ├── BudgetSection.tsx
    │   │   ├── SearchSection.tsx
    │   │   ├── FinalsSection.tsx
    │   │   ├── RunControls.tsx
    │   │   └── StatusBadge.tsx
    │   ├── monitor/
    │   │   ├── SessionHeader.tsx
    │   │   ├── LossChart.tsx
    │   │   ├── RunHistoryTable.tsx
    │   │   ├── GalleryAccordion.tsx
    │   │   └── Sparkline.tsx
    │   └── results/
    │       ├── Report.tsx
    │       ├── LedgerTable.tsx
    │       ├── ResultsGallery.tsx
    │       └── ConfigDiff.tsx
    ├── hooks/
    │   ├── useSession.ts            # GET /api/sessions/current + WS bridge
    │   ├── useLossStream.ts         # ring buffer + EMA derivation
    │   ├── useRuns.ts
    │   ├── useReport.ts
    │   ├── useJudgeStatus.ts
    │   ├── usePresets.ts
    │   ├── useHotkeys.ts
    │   └── useTheme.ts
    ├── lib/
    │   ├── api.ts                   # fetch wrapper (typed), throws on !ok
    │   ├── ws.ts                    # reconnecting socket
    │   ├── ema.ts                   # pure EMA function
    │   ├── lttb.ts                  # downsampling for >1k points (mirrors server)
    │   ├── format.ts                # duration / score / pct helpers
    │   └── queryClient.ts
    ├── stores/
    │   ├── session.ts               # Zustand: WS state, snapshot cache mirror
    │   ├── comparison.ts            # selected runs in Results
    │   └── ui.ts                    # smoothing, theme, sidebar collapsed
    ├── types/
    │   ├── api.ts                   # generated from OpenAPI
    │   └── domain.ts                # hand-written aliases
    └── styles/
        └── globals.css              # @import 'tailwindcss'
```

**Naming conventions:**
- Components: `PascalCase.tsx`, default-export the named component.
- Hooks: `useXxx.ts`, named exports only.
- Routes: lowercase with `$` prefix for params, per TanStack Router convention.
- Stores: `xxxStore` named export, file is the bare noun (`session.ts`).
- Tests (Phase 1+): co-located `Foo.test.tsx`, run via Vitest.

---

## 9. Type contract

Recommendation: **auto-generate from FastAPI's OpenAPI**, hand-write aliases on top.

Why: the orchestrator's dataclasses (`MonitorSnapshot`, `CandidateRow`, `LossSeries`) already have a clean shape; we mirror them as pydantic DTOs in the FastAPI router and let `openapi-typescript` produce `frontend/src/types/api.ts` on every backend change. Hand-writing keeps drifting (we've all seen it) and the orchestrator owns the shapes anyway.

Workflow:

```bash
# package.json
"scripts": {
  "gen:types": "openapi-typescript http://127.0.0.1:8000/openapi.json -o src/types/api.ts"
}
```

Run on backend change. CI step asserts no diff.

Hand-written domain aliases in `src/types/domain.ts`:

```ts
import type { components } from './api';

export type MonitorSnapshot = components['schemas']['MonitorSnapshotDTO'];
export type RunRow          = components['schemas']['RunRowDTO'];
export type LossSeries      = components['schemas']['LossSeriesDTO'];
export type GalleryGroup    = components['schemas']['GalleryGroupDTO'];
export type GalleryItem     = components['schemas']['GalleryItemDTO'];
export type JudgeReport     = components['schemas']['JudgeReportDTO'];
export type SessionConfig   = components['schemas']['SessionConfigDTO'];
export type Preset          = components['schemas']['PresetDTO'];
export type FieldSpec       = components['schemas']['FieldSpecDTO'];

// WS message envelope — hand-written; not part of the REST schema
export type WSMessage =
  | { type: 'snapshot'; data: MonitorSnapshot }
  | { type: 'log'; data: { line: string; level: string; ts: number } }
  | { type: 'run_completed'; data: RunRow }
  | { type: 'session_state'; data: { status: SessionStatus } }
  | { type: 'loss_frame'; data: { run_id: string; step: number; raw: number } }
  | { type: 'pong' };

export type SessionStatus = 'idle' | 'running' | 'stopping' | 'done' | 'error';
```

Reference shapes (what the DTOs will mirror — the Python source of truth is in `omnisteer_diffusion/ui/monitor.py`):

```ts
interface MonitorSnapshot {
  status: SessionStatus;
  setup_status: 'not started' | 'running' | 'done' | 'errored';
  output_dir: string | null;
  started_at: number | null;
  finished_at: number | null;
  elapsed_s: number;
  completed_runs: number;
  total_runs_target: number;
  progress_pct: number;
  current_run_id: string | null;
  current_run_steps_done: number | null;
  current_run_max_steps: number | null;
  current_loss: LossSeries | null;
  score_history: RunRow[];
  judge_summary: string;
  session_done: boolean;
  error_message: string | null;
}

interface RunRow {
  run_id: string;
  role: 'baseline' | 'candidate' | 'curated' | 'finalist' | 'setup';
  config_id: string;
  score: number | null;
  final_smoothed: number | null;
  slope: number | null;
  n_steps: number;
  duration_s: number;
  disqualified: string | null;
}

interface LossSeries {
  steps: number[];
  raw: number[];
  // smoothed is computed client-side; not sent over the wire
}

interface GalleryGroup {
  run_id: string;
  items: GalleryItem[];
  mtime: number;
}

interface GalleryItem {
  url: string;       // /files/runs/{run_id}/output/sample/foo.png
  filename: string;
  prompt_index: number;
  width?: number;
  height?: number;
}
```

---

## 10. Build / dev workflow

### Dev

```bash
# terminal 1: backend
uvicorn omnisteer_diffusion.api.app:app --reload --port 8000

# terminal 2: frontend
cd frontend && npm run dev          # vite on :5173
```

`vite.config.ts`:
```ts
server: {
  port: 5173,
  proxy: {
    '/api': 'http://127.0.0.1:8000',
    '/ws':  { target: 'ws://127.0.0.1:8000', ws: true },
    '/files': 'http://127.0.0.1:8000',
  },
},
```

Env: `VITE_API_BASE_URL` (defaults to empty = same-origin via proxy). Hot reload via Vite HMR; `react-refresh` preserves component state across edits.

### Build

```bash
cd frontend && npm run build        # → frontend/dist/
```

### Prod (single port)

FastAPI mounts the built bundle:

```py
app.mount("/files", StaticFiles(directory=str(SESSION_DIR), check_dir=False), name="files")
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="spa")
```

One process, one port (default 7860 to keep parity with the current Gradio launch command). The `/api` and `/ws` routes are matched first.

### Quality gates

- `npm run lint`  → `biome check src`
- `npm run format` → `biome format --write src`
- `npm run typecheck` → `tsc --noEmit`
- `npm run gen:types` → regenerate from OpenAPI

`lefthook.yml`:
```yml
pre-commit:
  parallel: true
  commands:
    biome:    { run: 'cd frontend && npm run lint',     glob: 'frontend/**/*.{ts,tsx}' }
    typecheck:{ run: 'cd frontend && npm run typecheck', glob: 'frontend/**/*.{ts,tsx}' }
```

Bundle-size report: `vite build --mode analyze` runs `rollup-plugin-visualizer` and writes `dist/stats.html`. Target: initial route ≤ 200 KB gzipped.

---

## 11. Phased migration plan

### Phase 0 — FastAPI alongside Gradio (S, ~1 day)

**Deliverables**
- New `omnisteer_diffusion/api/` package: `app.py`, `router.py`, `ws.py`, `dtos.py`.
- All REST endpoints from §3 wired to the existing `OrchestrationSession` singleton.
- WebSocket endpoint emits `snapshot` every 2 s while a session is running.
- `/files/runs/...` mount with path-traversal guard.
- OpenAPI exported at `/openapi.json`.
- Smoke tests: `pytest -k api` covering happy path + 409 conflict on double-start.

**Risks**
- The `OrchestrationSession` singleton is created at import time today — keep it. The FastAPI app reuses it.
- Logging handler currently hooks `omnisteer_diffusion` and `orchestrate` loggers from the session thread; that's fine — the WS endpoint just reads `session.snapshot()` and tails `state.log_tail`.

**Done when:** `curl :8000/api/sessions/current` returns the same data Gradio's Monitor tab renders, and `wscat -c ws://localhost:8000/ws/session` streams snapshots.

### Phase 1 — React frontend, side-by-side (M, ~5 days)

**Deliverables**
- Vite + React + TS skeleton in `frontend/` (replaces the static `index.html`; keep `serve.py` as `serve.py.old` for one release).
- All 4 pages implemented at parity with Gradio (Setup, Run, Monitor, Results).
- TanStack Query + WS bridge functioning (Monitor live updates, no manual refresh button).
- Sticky run-controls bar.
- Keyboard shortcuts (`s`, `Esc`, `r`, `Cmd+K`).
- Comparison mode in Results gallery (the headline new capability).
- Deep links (`/results?run=cand-007`).
- Single-port serve via `npm run build` + FastAPI `StaticFiles` mount.
- `python -m omnisteer_diffusion.ui.app` still works; new `python -m omnisteer_diffusion.api.app` serves React.

**Risks**
- WS reconnect on subprocess death — mitigate with a dedicated `useEffect` that re-fetches `/api/sessions/current` on every reconnect and a visible reconnect badge.
- Gallery image counts can hit 1000+ — lazy-load and cap initial render at 24 per group, "Show more" expands.
- Smoothing slider over 5000 raw points — memoise EMA; if it drops below 60 fps, downsample at the WS edge with LTTB.

**Done when:** the user can run a full session through the React UI without touching `:7860`. Gradio still works as fallback. README updated with both launch commands.

### Phase 2 — Deprecate Gradio (S, ~½ day)

**Deliverables**
- Remove `omnisteer_diffusion/ui/` (`app.py`, `monitor.py`'s UI helpers stay — `build_snapshot`, `gallery_groups` are now used by the FastAPI DTO mappers).
- Drop `gradio>=4.40` from `pyproject.toml`.
- `omnisteer-diffusion-ui` console script points at the FastAPI launcher.
- `docs/UI_GUIDE.md` rewritten for the React UI (keyboard shortcuts, comparison mode, theme toggle).
- Final smoke test — `tests/ui/` becomes `tests/api/` covering all endpoints.

**Risks**
- Anyone relying on `app.launch(share=True)` Gradio tunnels — document the alternative (Cloudflare Tunnel or Tailscale Funnel) in the README.
- The `serve.py` static dashboard at `:3000` should also be removed; surface a single canonical entry point.

**Done when:** `gradio` is no longer in the dep tree and `python -m pip uninstall gradio` doesn't break anything.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| WS reconnection edge cases when training subprocess dies (server still up, session in `error` state, client unaware). | On every WS open + every interval, client issues `GET /api/sessions/current` as the source of truth; WS messages are deltas, REST is authoritative. |
| Image-serving security — FastAPI exposing arbitrary disk paths via `/files/runs/...`. | Whitelist file extensions; resolve symlinks; assert resolved path is under `session.output_dir / "runs"`; reject `..`; never serve from outside the active session. |
| First-load bundle size. | Vite code-splits routes by default; keep Recharts on the Monitor route only (lazy-load), keep Shiki off the critical path entirely (used only in Results' diff view). Target ≤ 200 KB gzip on `/setup`; CI fails build if exceeded. |
| State drift between TanStack Query cache and server truth (`completed_runs` jitter, ledger row missing). | `staleTime: 0` for active session queries, `refetchOnReconnect: true`, WS `run_completed` triggers `queryClient.invalidateQueries(['runs'])`. The REST snapshot is always re-fetched on tab focus. |
| Long tfevents (5000+ points) tank chart fps. | Server-side LTTB downsamples to ≤1000 points before the WS push; raw series available via `GET /loss?since=...` for users who want the unsmoothed full data; `<LineChart>` has `isAnimationActive={false}` after first paint. |
| FastAPI thread-safety with `OrchestrationSession.snapshot()` polled at 2 Hz from the WS task while the session thread mutates state. | Already protected by `RLock` (see `ui/session.py:91`). The new WS endpoint runs on the asyncio loop; wrap the snapshot call in `asyncio.to_thread` to avoid blocking the loop on lock contention. |
| Tauri wrap (future) needs file-system access for path pickers — can't reuse the React-only build verbatim. | Keep all file-input handling behind a `useFilePicker` hook that branches: `window.__TAURI__ ? tauri.dialog.open() : <Input>`. Phase 1 does the web fallback; Tauri is a Phase 3 concern. |

---

*End of plan. Update on backend DTO change; regenerate `src/types/api.ts` from `/openapi.json` whenever the FastAPI router changes.*
