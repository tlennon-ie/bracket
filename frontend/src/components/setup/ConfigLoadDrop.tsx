// ConfigLoadDrop — drag-and-drop or pick a bracket config JSON to
// populate every field on the Setup + Run pages at once. Pairs with the
// "Export config" button to round-trip a complete bracket setup.
//
// Schema is the same as `ConfigBundle` from /api/config — a small
// envelope around a `StartSessionRequest`. Validation is best-effort
// here (we set fields one-by-one) and authoritative on the backend
// (POST /api/config/validate returns Pydantic field-level errors).

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { useSessionConfigStore } from "@/store/sessionConfig";
import type { ConfigBundle, StartSessionRequest } from "@/types/domain";
import { Download, FileCheck2, FileWarning, Upload } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";

type FieldKey = keyof StartSessionRequest;

const KNOWN_FIELDS: readonly FieldKey[] = [
	"family",
	"training_type",
	"dataset_toml",
	"output_dir",
	"sample_prompts",
	"resume",
	"images_per_dataset",
	"vram_gb",
	"budget",
	"max_steps",
	"wall_secs",
	"seeds",
	"search_method",
	"optuna_startup",
	"n_curated",
	"finals_top_k",
	"finals_max_steps",
	"finals_seeds",
	"judge_method",
	"judge_base_url",
	"judge_model",
	"judge_loss_weight",
	"judge_sample_weight",
	"judge_disable_thinking",
	"preset_field_values",
];

function isPlainObject(v: unknown): v is Record<string, unknown> {
	return typeof v === "object" && v !== null && !Array.isArray(v);
}

function extractRequest(raw: unknown): StartSessionRequest | string {
	if (!isPlainObject(raw)) return "config must be a JSON object";
	// Accept either the bare request or the bundle envelope.
	const candidate: unknown =
		"request" in raw && isPlainObject(raw.request) ? raw.request : raw;
	if (!isPlainObject(candidate)) return "config has no 'request' object";
	const missing = KNOWN_FIELDS.filter((k) => !(k in candidate));
	if (missing.length > KNOWN_FIELDS.length - 5) {
		return `config doesn't look like a bracket bundle (missing ${missing.length} expected fields)`;
	}
	return candidate as unknown as StartSessionRequest;
}

interface Props {
	className?: string;
}

export function ConfigLoadDrop({ className }: Props) {
	const set = useSessionConfigStore((s) => s.set);
	const config = useSessionConfigStore((s) => s.config);
	const setFamilyType = useSessionConfigStore((s) => s.setFamilyType);
	const [hover, setHover] = useState(false);
	const [loadedName, setLoadedName] = useState<string | null>(null);
	const [error, setError] = useState<string | null>(null);
	const fileInputRef = useRef<HTMLInputElement | null>(null);

	const applyConfig = useCallback(
		(req: StartSessionRequest, sourceName: string) => {
			// Family + training_type need their combined setter so preset
			// field values reset cleanly when the family changes.
			setFamilyType(req.family, req.training_type);
			for (const key of KNOWN_FIELDS) {
				if (key === "family" || key === "training_type") continue;
				const value = req[key];
				if (value === undefined) continue;
				// biome-ignore lint/suspicious/noExplicitAny: index over union
				set(key as FieldKey, value as any);
			}
			setLoadedName(sourceName);
			setError(null);
			toast.success("Config loaded", {
				description: `${sourceName} — ${req.family || "?"} / ${req.training_type || "?"}, budget ${req.budget}, seeds ${req.seeds}`,
			});
		},
		[set, setFamilyType],
	);

	const handleFile = useCallback(
		async (file: File) => {
			let text: string;
			try {
				text = await file.text();
			} catch (e) {
				setError(e instanceof Error ? e.message : "could not read file");
				return;
			}
			let parsed: unknown;
			try {
				parsed = JSON.parse(text);
			} catch (e) {
				setError(e instanceof Error ? e.message : "invalid JSON");
				return;
			}
			const result = extractRequest(parsed);
			if (typeof result === "string") {
				setError(result);
				return;
			}
			applyConfig(result, file.name);
		},
		[applyConfig],
	);

	const handleExport = useCallback(() => {
		const bundle: ConfigBundle = {
			bracket_version: "0.1.0",
			saved_at: Date.now() / 1000,
			name: "bracket-session-config",
			request: config,
		};
		const blob = new Blob([JSON.stringify(bundle, null, 2)], {
			type: "application/json",
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
		a.href = url;
		a.download = `bracket-config-${stamp}.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
		toast.success("Config exported", {
			description: `${a.download} — re-import with the drop zone above.`,
		});
	}, [config]);

	return (
		<Card
			onDragOver={(e) => {
				e.preventDefault();
				setHover(true);
			}}
			onDragLeave={() => setHover(false)}
			onDrop={(e) => {
				e.preventDefault();
				setHover(false);
				const f = e.dataTransfer.files[0];
				if (f) void handleFile(f);
			}}
			className={cn(
				"border-dashed transition-colors",
				hover && "border-primary bg-primary/5",
				className,
			)}
		>
			<CardContent className="flex items-start gap-3 p-4">
				<Upload className="h-5 w-5 mt-0.5 text-muted-foreground shrink-0" />
				<div className="flex-1 min-w-0">
					<div className="flex flex-wrap items-center justify-between gap-2">
						<div>
							<p className="text-sm font-medium">Load config</p>
							<p className="text-xs text-muted-foreground">
								Drop a bracket config JSON to populate every field on Setup +
								Run. Or{" "}
								<button
									type="button"
									className="underline underline-offset-2 hover:text-foreground"
									onClick={() => fileInputRef.current?.click()}
								>
									pick a file
								</button>
								.
							</p>
						</div>
						<button
							type="button"
							onClick={handleExport}
							className="inline-flex items-center gap-1.5 text-xs text-foreground border border-border rounded-md px-2.5 py-1 hover:bg-accent"
						>
							<Download className="h-3.5 w-3.5" />
							Export config
						</button>
					</div>
					<input
						ref={fileInputRef}
						type="file"
						accept="application/json,.json"
						className="hidden"
						onChange={(e) => {
							const f = e.target.files?.[0];
							if (f) void handleFile(f);
							e.target.value = "";
						}}
					/>
					{loadedName && (
						<div className="mt-3 flex items-center gap-2 text-xs">
							<FileCheck2 className="h-4 w-4 text-success" />
							<code className="font-mono-tight">{loadedName}</code>
							<span className="text-muted-foreground">
								— all setup + run fields populated.
							</span>
						</div>
					)}
					{error && (
						<div className="mt-3 flex items-center gap-2 text-xs text-destructive">
							<FileWarning className="h-4 w-4" />
							{error}
						</div>
					)}
				</div>
			</CardContent>
		</Card>
	);
}
