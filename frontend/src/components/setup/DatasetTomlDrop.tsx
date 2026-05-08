// Drag-and-drop dataset TOML preview.
//
// Browser can't read arbitrary disk paths — this only previews a file the
// user dropped. The path the backend receives is still typed manually
// (or, in a future Tauri build, picked via tauri.dialog.open).

import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { FileCheck2, FileWarning, Upload } from "lucide-react";
import { useCallback, useState } from "react";
import { parse as parseToml } from "smol-toml";

interface ParsedSummary {
	fileName: string;
	bucketCount: number;
	imageCount: number;
	resolutions: string[];
	raw: string;
}

interface Props {
	onPreview?: (summary: ParsedSummary) => void;
}

function summarizeToml(
	name: string,
	raw: string,
): ParsedSummary | { error: string } {
	let parsed: unknown;
	try {
		parsed = parseToml(raw);
	} catch (e) {
		return { error: e instanceof Error ? e.message : "invalid TOML" };
	}
	// Heuristic: count `[[datasets]]` entries and unique resolutions inside them.
	const datasets =
		(parsed && typeof parsed === "object" && "datasets" in parsed
			? ((parsed as { datasets: unknown }).datasets as unknown[])
			: []) ?? [];
	let imageCount = 0;
	const resolutions = new Set<string>();
	for (const d of datasets) {
		if (!d || typeof d !== "object") continue;
		const obj = d as Record<string, unknown>;
		const subsets = (obj.subsets ?? obj.image_dirs ?? []) as unknown[];
		if (Array.isArray(subsets)) {
			for (const s of subsets) {
				if (s && typeof s === "object") {
					const sb = s as Record<string, unknown>;
					if (typeof sb.num_repeats === "number")
						imageCount += Math.max(1, sb.num_repeats);
				}
			}
		}
		if (typeof obj.resolution === "string") resolutions.add(obj.resolution);
		else if (Array.isArray(obj.resolution))
			resolutions.add(obj.resolution.map(String).join("×"));
	}
	return {
		fileName: name,
		bucketCount: Array.isArray(datasets) ? datasets.length : 0,
		imageCount,
		resolutions: Array.from(resolutions),
		raw,
	};
}

export function DatasetTomlDrop({ onPreview }: Props) {
	const [hover, setHover] = useState(false);
	const [summary, setSummary] = useState<ParsedSummary | null>(null);
	const [error, setError] = useState<string | null>(null);

	const handleFile = useCallback(
		async (file: File) => {
			const text = await file.text();
			const result = summarizeToml(file.name, text);
			if ("error" in result) {
				setError(result.error);
				setSummary(null);
				return;
			}
			setError(null);
			setSummary(result);
			onPreview?.(result);
		},
		[onPreview],
	);

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
			)}
		>
			<CardContent className="flex items-start gap-3 p-4">
				<Upload className="h-5 w-5 mt-0.5 text-muted-foreground shrink-0" />
				<div className="flex-1 min-w-0">
					<p className="text-sm font-medium">Drop a dataset TOML to preview</p>
					<p className="text-xs text-muted-foreground">
						Bucket counts and resolutions parse client-side. The backend still
						reads from the path you type below.
					</p>
					{summary && (
						<div className="mt-3 flex items-center gap-2 text-xs">
							<FileCheck2 className="h-4 w-4 text-success" />
							<code className="font-mono-tight">{summary.fileName}</code>
							<span className="text-muted-foreground">·</span>
							<span>
								{summary.bucketCount} bucket
								{summary.bucketCount === 1 ? "" : "s"}
							</span>
							{summary.resolutions.length > 0 && (
								<>
									<span className="text-muted-foreground">·</span>
									<span className="font-mono-tight">
										{summary.resolutions.join(" / ")}
									</span>
								</>
							)}
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
