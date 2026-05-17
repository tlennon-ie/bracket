// Hand-written mirror of `bracket/api/schemas.py`.
//
// Keep these in sync with the Pydantic models. After backend changes:
//   `npm run gen:types` overwrites src/types/api.ts from /openapi.json.
//   This file is the human-friendly aliases; treat api.ts as source of truth.

export type SessionStatus = "idle" | "running" | "stopping" | "done" | "error";
export type SetupStatus = "not started" | "running" | "done" | "errored";
export type RunRole =
	| "baseline"
	| "candidate"
	| "curated"
	| "finalist"
	| "setup"
	| string;

export interface FieldSpec {
	name: string;
	label: string;
	default: string;
	required: boolean;
	kind: string;
	help: string;
	target: string;
}

export interface ModelFamily {
	name: string;
	label: string;
}

export interface TrainingType {
	name: string;
	label: string;
}

export interface Preset {
	id: string;
	family: string;
	training_type: string;
	display_name: string;
	notes: string;
	needs_pre_cache: boolean;
	fields: FieldSpec[];
	session_fields: FieldSpec[];
}

export interface CandidateRow {
	run_id: string;
	role: RunRole;
	config_id: string;
	score: number | null;
	final_smoothed: number | null;
	slope: number | null;
	n_steps: number;
	duration_s: number;
	disqualified: string | null;
}

export interface LossSeries {
	steps: number[];
	raw: number[];
	smoothed: number[];
	// Per-step gradient norm. Either empty (trainer doesn't log
	// grad_norm) or the same length as `steps`, with `null` entries on
	// steps where the trainer didn't emit a value.
	grad_norms?: Array<number | null>;
}

export interface MonitorSnapshot {
	session_status: SessionStatus;
	output_dir: string | null;
	elapsed_s: number;
	error_message: string | null;
	status_line: string;
	progress_pct: number;
	completed_runs: number;
	total_runs_target: number;
	current_run_id: string | null;
	current_run_steps_done: number | null;
	current_run_max_steps: number | null;
	current_loss: LossSeries | null;
	score_history: CandidateRow[];
	setup_status: SetupStatus;
	judge_summary: string;
	session_done: boolean;
	current_steps_per_sec: number | null;
	killed_by_pruner: number;
	running_runs: number;
	ts: number;
}

export interface GalleryItem {
	path: string;
	url: string;
	caption: string;
	run_id: string;
}

export interface GalleryGroup {
	run_id: string;
	items: GalleryItem[];
	mtime: number;
}

export interface RunDetail {
	run_id: string;
	role: RunRole;
	config_id: string;
	score: number | null;
	n_steps: number;
	duration_s: number;
	disqualified: string | null;
	config: Record<string, string>;
	score_components: Record<string, number>;
	judge_report: Record<string, number>;
	run_dir: string;
	log_path: string | null;
	sample_dir: string | null;
}

export interface RunLogChunk {
	exists: boolean;
	content: string;
	offset: number;
	next_offset: number;
	total_size: number;
	truncated_to_tail: boolean;
}

export interface ReportPayload {
	path: string;
	content: string;
}

export interface JudgeStatus {
	configured: boolean;
	summary: string;
}

export interface HealthPayload {
	status: string;
	version: string;
}

// Body for POST /api/session/start — every field on bracket/api/schemas.py::StartSessionRequest.
export interface StartSessionRequest {
	family: string;
	training_type: string;
	dataset_toml: string;
	output_dir: string;
	sample_prompts: string;
	sample_prompts_ood?: string | null;
	resume: string;
	images_per_dataset: number;
	vram_gb: number | null;
	budget: number;
	max_steps: number;
	wall_secs: number;
	seeds: number;
	search_method: string;
	optuna_startup: number;
	n_curated: number;
	finals_top_k: number;
	finals_max_steps: number;
	finals_seeds: number;
	judge_method: string;
	judge_base_url: string;
	judge_model: string;
	judge_loss_weight: number;
	judge_sample_weight: number;
	judge_disable_thinking: boolean;
	judge_n_samples?: number;
	enable_clip_iqa_gate?: boolean;
	clip_iqa_dq_threshold?: number;
	lr_min: number | null;
	lr_max: number | null;
	batch_size_min: number | null;
	batch_size_max: number | null;
	gradient_checkpointing_mode: "on" | "off" | "search" | null;
	use_history_priors?: boolean;
	enable_divergence_killer: boolean;
	enable_two_rung_asha: boolean;
	objectives: string[];
	enable_pairwise_finals: boolean;
	preset_field_values: Record<string, string>;
}

export interface StartSessionResponse {
	status: "started" | "conflict" | "bad_request" | string;
	message: string;
	session_id: string | null;
	output_dir: string | null;
}

export interface StopSessionResponse {
	stopped: boolean;
	message: string;
}

export interface TrainingConfigExport {
	run_id: string;
	role: string;
	family: string;
	training_type: string;
	score: number | null;
	config: Record<string, string>;
	source_dataset_toml: string | null;
	sample_prompts: string | null;
	preset_field_values: Record<string, string>;
	notes: string;
}

export interface PromoteRunRequest {
	output_dir?: string;
	max_steps: number;
	save_every_n_steps: number;
	save_state: boolean;
	sample_every_n_steps: number | null;
	resume_from: string;
	full_dataset_toml: string;
}

export interface PromoteRunResponse {
	status: "started" | "conflict" | "bad_request" | string;
	message: string;
	promoted_run_id: string | null;
	output_dir: string | null;
}

export interface ConfigBundle {
	bracket_version: string;
	saved_at: number;
	name: string;
	request: StartSessionRequest;
}

export interface UpdateStatus {
	current_version: string;
	latest_version: string | null;
	update_available: boolean;
	release_url: string | null;
	release_notes: string | null;
	checked_at: number;
	error: string | null;
}

export interface TriggerUpdateResponse {
	spawned: boolean;
	message: string;
	script: string | null;
	log_path: string | null;
}
