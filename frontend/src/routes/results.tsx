import { ComparisonGrid } from "@/components/results/ComparisonGrid";
import { ConfigDiff } from "@/components/results/ConfigDiff";
import { ReportViewer } from "@/components/results/ReportViewer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Skeleton } from "@/components/ui/skeleton";
import { useGallery } from "@/hooks/useGallery";
import { useRegenerateReport, useReport } from "@/hooks/useReport";
import { useRuns } from "@/hooks/useRuns";
import { api } from "@/lib/api";
import { formatDuration, formatScore, shortenRunId } from "@/lib/format";
import { useUIStore } from "@/store/ui";
import { useQueries } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { FileText, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { z } from "zod";

const searchSchema = z.object({
	run: z.string().optional(),
	compare: z.string().optional(),
});

export const Route = createFileRoute("/results")({
	validateSearch: searchSchema,
	component: ResultsPage,
});

function ResultsPage() {
	const search = Route.useSearch();
	const navigate = useNavigate({ from: Route.fullPath });
	const compareRunIds = useUIStore((s) => s.compareRunIds);
	const setCompareRuns = useUIStore((s) => s.setCompareRuns);
	const toggleCompareRun = useUIStore((s) => s.toggleCompareRun);
	const clearCompareRuns = useUIStore((s) => s.clearCompareRuns);

	const report = useReport();
	const runs = useRuns();
	const gallery = useGallery();
	const regenerate = useRegenerateReport();

	// Hydrate URL → store on first render only.
	const hydrated = useRef(false);
	if (!hydrated.current) {
		hydrated.current = true;
		const fromUrl: string[] = [];
		if (search.run) fromUrl.push(search.run);
		if (search.compare)
			fromUrl.push(...search.compare.split(",").filter(Boolean));
		if (fromUrl.length > 0) setCompareRuns(fromUrl);
	}

	// Mirror store → URL.
	useEffect(() => {
		const [first, ...rest] = compareRunIds;
		void navigate({
			search: {
				run: first,
				compare: rest.length > 0 ? rest.join(",") : undefined,
			},
			replace: true,
		});
	}, [compareRunIds, navigate]);

	// Load full RunDetail for each compared run (for the diff view).
	const detailQueries = useQueries({
		queries: compareRunIds.map((id) => ({
			queryKey: ["run-detail", id],
			queryFn: () => api.getRun(id),
			staleTime: 30_000,
		})),
	});
	const details = detailQueries
		.map((q) => q.data)
		.filter((d): d is NonNullable<typeof d> => !!d);

	const sortedRuns = useMemo(() => {
		return (runs.data ?? [])
			.slice()
			.sort(
				(a, b) =>
					(a.score ?? Number.POSITIVE_INFINITY) -
					(b.score ?? Number.POSITIVE_INFINITY),
			);
	}, [runs.data]);

	return (
		<div className="flex flex-col gap-6">
			<header className="flex flex-wrap items-end justify-between gap-3 no-print">
				<div className="flex flex-col gap-1">
					<h1 className="text-3xl font-semibold tracking-tight font-display">
						Results
					</h1>
					<p className="text-sm text-muted-foreground">
						Read the report. Diff configs. Compare samples side-by-side.
					</p>
				</div>
				<div className="flex items-center gap-2">
					<Button
						variant="outline"
						size="sm"
						onClick={() => regenerate.mutate()}
						disabled={regenerate.isPending}
						className="gap-1.5"
					>
						<RefreshCw
							className={
								regenerate.isPending
									? "h-3.5 w-3.5 animate-spin"
									: "h-3.5 w-3.5"
							}
						/>
						{regenerate.isPending ? "Regenerating…" : "Regenerate report"}
					</Button>
					<Button
						variant="outline"
						size="sm"
						onClick={() => window.print()}
						className="gap-1.5"
					>
						<FileText className="h-3.5 w-3.5" />
						Print
					</Button>
				</div>
			</header>

			<Card>
				<CardHeader>
					<CardTitle>Report</CardTitle>
				</CardHeader>
				<CardContent>
					{report.isLoading ? (
						<div className="flex flex-col gap-2">
							<Skeleton className="h-6 w-1/2" />
							<Skeleton className="h-4 w-3/4" />
							<Skeleton className="h-4 w-2/3" />
						</div>
					) : (
						<ReportViewer markdown={report.data ?? null} />
					)}
				</CardContent>
			</Card>

			<Card className="no-print">
				<CardHeader className="flex-row items-center justify-between space-y-0">
					<CardTitle>Ledger</CardTitle>
					{compareRunIds.length > 0 && (
						<div className="flex items-center gap-2">
							<Badge variant="muted">{compareRunIds.length} selected</Badge>
							<Button
								variant="ghost"
								size="sm"
								onClick={clearCompareRuns}
								className="gap-1"
							>
								<Trash2 className="h-3 w-3" />
								Clear
							</Button>
						</div>
					)}
				</CardHeader>
				<CardContent>
					{runs.isLoading ? (
						<Skeleton className="h-32 w-full" />
					) : sortedRuns.length === 0 ? (
						<p className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg">
							No runs in the ledger yet.
						</p>
					) : (
						<div className="overflow-x-auto">
							<table className="w-full text-sm border-collapse">
								<thead>
									<tr className="text-[10px] text-muted-foreground uppercase tracking-wider border-b border-border">
										<th
											className="text-left font-medium py-2 pr-3 w-8"
											aria-label="Compare"
										>
											✓
										</th>
										<th className="text-left font-medium py-2 pr-3">Run</th>
										<th className="text-left font-medium py-2 pr-3">Role</th>
										<th className="text-left font-medium py-2 pr-3">Config</th>
										<th className="text-right font-medium py-2 pr-3">Score</th>
										<th className="text-right font-medium py-2 pr-3">Steps</th>
										<th className="text-right font-medium py-2 pr-3">
											Duration
										</th>
									</tr>
								</thead>
								<tbody>
									{sortedRuns.map((r) => {
										const checked = compareRunIds.includes(r.run_id);
										return (
											<tr
												key={r.run_id}
												className="border-b border-border/60 hover:bg-accent/30"
											>
												<td className="py-2 pr-3">
													<Checkbox
														checked={checked}
														onCheckedChange={() => toggleCompareRun(r.run_id)}
														disabled={!checked && compareRunIds.length >= 3}
														aria-label={`Compare ${r.run_id}`}
													/>
												</td>
												<td className="py-2 pr-3">
													<code className="font-mono-tight text-xs">
														{shortenRunId(r.run_id, 26)}
													</code>
												</td>
												<td className="py-2 pr-3">
													<Badge
														variant="outline"
														className="capitalize text-[10px]"
													>
														{r.role}
													</Badge>
												</td>
												<td className="py-2 pr-3">
													<code className="font-mono-tight text-xs text-muted-foreground">
														{shortenRunId(r.config_id, 14)}
													</code>
												</td>
												<td className="py-2 pr-3 text-right font-mono-tight">
													{formatScore(r.score)}
												</td>
												<td className="py-2 pr-3 text-right font-mono-tight text-muted-foreground">
													{r.n_steps.toLocaleString()}
												</td>
												<td className="py-2 pr-3 text-right font-mono-tight text-muted-foreground">
													{formatDuration(r.duration_s)}
												</td>
											</tr>
										);
									})}
								</tbody>
							</table>
						</div>
					)}
				</CardContent>
			</Card>

			{compareRunIds.length >= 2 && (
				<Card className="no-print">
					<CardHeader>
						<CardTitle>Config diff</CardTitle>
					</CardHeader>
					<CardContent>
						<ConfigDiff runs={details} />
					</CardContent>
				</Card>
			)}

			{compareRunIds.length > 0 && (
				<Card>
					<CardHeader>
						<CardTitle>Comparison gallery</CardTitle>
					</CardHeader>
					<CardContent>
						<ComparisonGrid
							groups={gallery.data ?? []}
							runIds={compareRunIds}
						/>
					</CardContent>
				</Card>
			)}
		</div>
	);
}
