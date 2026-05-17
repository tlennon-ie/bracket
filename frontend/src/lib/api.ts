// Centralised fetch wrapper. Throws ApiError on non-2xx; surfaces a toast.
//
// Same-origin in prod (FastAPI mounts the SPA), Vite dev-proxy in dev.

import type {
	CandidateRow,
	ConfigBundle,
	GalleryGroup,
	HealthPayload,
	JudgeStatus,
	LossSeries,
	ModelFamily,
	MonitorSnapshot,
	Preset,
	PromoteRunRequest,
	PromoteRunResponse,
	ReportPayload,
	RunDetail,
	RunLogChunk,
	StartSessionRequest,
	StartSessionResponse,
	StopSessionResponse,
	TrainingConfigExport,
	TrainingType,
	TriggerUpdateResponse,
	UpdateStatus,
} from "@/types/domain";
import { toast } from "sonner";

const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
	status: number;
	body: unknown;
	constructor(message: string, status: number, body: unknown) {
		super(message);
		this.name = "ApiError";
		this.status = status;
		this.body = body;
	}
}

interface RequestOpts {
	/** Suppress automatic toast on error (useful when caller renders inline). */
	silent?: boolean;
	/** Don't throw on the listed status codes; return the body normally. */
	acceptStatuses?: number[];
}

async function request<T>(
	path: string,
	init: RequestInit = {},
	opts: RequestOpts = {},
): Promise<T> {
	const url = `${BASE}${path}`;
	let res: Response;
	try {
		res = await fetch(url, {
			...init,
			headers: {
				"Content-Type": "application/json",
				Accept: "application/json",
				...(init.headers ?? {}),
			},
		});
	} catch (e) {
		const msg = e instanceof Error ? e.message : "network error";
		if (!opts.silent) {
			toast.error("Bracket API unreachable", {
				description: `${msg}. Is bracket-serve running on port 8000?`,
			});
		}
		throw new ApiError(`network: ${msg}`, 0, null);
	}
	const accepted = opts.acceptStatuses ?? [];
	if (!res.ok && !accepted.includes(res.status)) {
		let detail: unknown = null;
		try {
			detail = await res.json();
		} catch {
			// ignore
		}
		const message =
			(detail && typeof detail === "object" && "detail" in detail
				? String((detail as { detail: unknown }).detail)
				: `${res.status} ${res.statusText}`) || `${res.status}`;
		if (!opts.silent) {
			toast.error(`API ${res.status}`, { description: message });
		}
		throw new ApiError(message, res.status, detail);
	}
	// Some endpoints (text/markdown) return non-JSON.
	const ct = res.headers.get("content-type") ?? "";
	if (ct.includes("application/json")) {
		return (await res.json()) as T;
	}
	return (await res.text()) as unknown as T;
}

// ─── REST surface ────────────────────────────────────────────────────────────

export const api = {
	health: (): Promise<HealthPayload> => request("/api/health"),

	// Presets — cascading dropdown.
	listFamilies: (): Promise<ModelFamily[]> => request("/api/presets/families"),
	listTrainingTypes: (family: string): Promise<TrainingType[]> =>
		request(`/api/presets/families/${encodeURIComponent(family)}/types`),
	getPreset: (family: string, type: string): Promise<Preset> =>
		request(
			`/api/presets/${encodeURIComponent(family)}/${encodeURIComponent(type)}`,
		),

	// Session.
	getSnapshot: (silent = false): Promise<MonitorSnapshot> =>
		request("/api/session", {}, { silent }),
	startSession: (req: StartSessionRequest): Promise<StartSessionResponse> =>
		request(
			"/api/session/start",
			{ method: "POST", body: JSON.stringify(req) },
			{ acceptStatuses: [400, 409] },
		),
	stopSession: (): Promise<StopSessionResponse> =>
		request("/api/session/stop", { method: "POST" }),

	// Runs.
	listRuns: (): Promise<CandidateRow[]> => request("/api/runs"),
	getRun: (runId: string): Promise<RunDetail> =>
		request(`/api/runs/${encodeURIComponent(runId)}`),
	getRunLog: (
		runId: string,
		offset = 0,
		silent = true,
	): Promise<RunLogChunk> =>
		request(
			`/api/runs/${encodeURIComponent(runId)}/log?offset=${offset}`,
			{},
			{ silent, acceptStatuses: [404] },
		),
	// The ema_alpha query param is documented as 0.001..1.0 but the chart
	// re-smooths client-side, so the server smoothing is just a fallback.
	getRunLoss: (runId: string, emaAlpha = 0.05): Promise<LossSeries> =>
		request(
			`/api/runs/${encodeURIComponent(runId)}/loss?ema_alpha=${emaAlpha}`,
			{},
			{ acceptStatuses: [404] },
		),
	exportTrainingConfig: (runId: string): Promise<TrainingConfigExport> =>
		request(`/api/runs/${encodeURIComponent(runId)}/training-config`),
	promoteRun: (
		runId: string,
		body: PromoteRunRequest,
	): Promise<PromoteRunResponse> =>
		request(
			`/api/runs/${encodeURIComponent(runId)}/promote`,
			{ method: "POST", body: JSON.stringify(body) },
			{ acceptStatuses: [400, 404, 409] },
		),

	// Config import / export.
	exportConfig: (): Promise<ConfigBundle> =>
		request("/api/config", {}, { acceptStatuses: [404] }),
	validateConfig: (bundle: { request: StartSessionRequest }): Promise<ConfigBundle> =>
		request(
			"/api/config/validate",
			{ method: "POST", body: JSON.stringify(bundle) },
		),

	// Gallery.
	getGallery: (): Promise<GalleryGroup[]> => request("/api/gallery"),

	// Report.
	getReport: (): Promise<string> =>
		request("/api/report", {}, { acceptStatuses: [404] }),
	regenerateReport: (): Promise<ReportPayload> =>
		request("/api/report/regenerate", { method: "POST" }),

	// Judge.
	getJudgeStatus: (): Promise<JudgeStatus> => request("/api/judge/status"),

	// Updater.
	checkForUpdate: (force = false): Promise<UpdateStatus> =>
		request(
			`/api/update/check${force ? "?force=true" : ""}`,
			{},
			{ silent: true },
		),
	applyUpdate: (): Promise<TriggerUpdateResponse> =>
		request("/api/update/apply", { method: "POST" }),
};

export type ApiClient = typeof api;
