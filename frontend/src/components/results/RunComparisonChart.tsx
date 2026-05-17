// RunComparisonChart — loss curves for one or many selected runs.
//
// One checkbox in the Results-page ledger = one curve. Each run gets a
// stable colour so picking the same run twice never reshuffles the
// palette. We smooth client-side using the global EMA-alpha slider so
// the user can tune all selected curves together without refetching.
//
// Raw points are drawn at low opacity behind the smoothed line so the
// noise floor is visible without dominating the chart.

import { LossChartLegend } from "@/components/results/LossChartLegend";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { applyEMA } from "@/lib/ema";
import { lttb } from "@/lib/lttb";
import { useUIStore } from "@/store/ui";
import type { LossSeries } from "@/types/domain";
import { useQueries } from "@tanstack/react-query";
import { useMemo } from "react";
import {
	CartesianGrid,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

interface Props {
	runIds: string[];
}

// Capped to keep Recharts responsive when several long runs are overlaid.
const MAX_RENDER_POINTS = 800;

// Distinct, accessible-on-dark hues. First slot is the bracket accent so
// a single-run comparison matches the live Monitor chart's colour.
const RUN_COLORS = [
	"var(--accent-bracket)",
	"#22d3ee", // cyan-400
	"#f59e0b", // amber-500
	"#a78bfa", // violet-400
	"#34d399", // emerald-400
	"#f472b6", // pink-400
	"#fb923c", // orange-400
	"#60a5fa", // blue-400
] as const;

export function colorForIndex(i: number): string {
	return RUN_COLORS[i % RUN_COLORS.length] ?? RUN_COLORS[0];
}

interface ChartRow {
	step: number;
	[seriesKey: string]: number | null;
}

export function RunComparisonChart({ runIds }: Props) {
	const smoothing = useUIStore((s) => s.smoothing);

	const queries = useQueries({
		queries: runIds.map((id) => ({
			queryKey: ["run-loss", id],
			queryFn: () => api.getRunLoss(id),
			staleTime: 30_000,
			// Tolerate 404s — pre-step-0 runs or disqualified-no-tfevents
			// rows shouldn't blank the whole chart for the rest.
			retry: false,
		})),
	});

	const isLoading = queries.some((q) => q.isLoading);

	// Per-run smoothed values keyed by run_id.
	const seriesByRun = useMemo(() => {
		const out: Record<string, { steps: number[]; smoothed: number[]; raw: number[] }> = {};
		runIds.forEach((id, i) => {
			const data = queries[i]?.data as LossSeries | null | undefined;
			// 404s return a {detail: ...} blob — guard against the missing
			// ``steps`` array so one DQ'd run doesn't blank the whole chart.
			if (!data || !Array.isArray(data.steps) || data.steps.length === 0) return;
			const smoothed = applyEMA(data.raw, smoothing);
			out[id] = { steps: data.steps, raw: data.raw, smoothed };
		});
		return out;
	}, [runIds, queries, smoothing]);

	// Merge into Recharts rows keyed by step. Recharts needs a single array
	// where each row carries every series' value for the same X — we leave
	// missing values as ``null`` so runs of different lengths still plot.
	const data = useMemo<ChartRow[]>(() => {
		const allSteps = new Set<number>();
		for (const id of runIds) {
			const s = seriesByRun[id];
			if (!s) continue;
			for (const step of s.steps) allSteps.add(step);
		}
		if (allSteps.size === 0) return [];

		const sortedSteps = Array.from(allSteps).sort((a, b) => a - b);

		// Step → row-index in each run's series for O(1) lookup.
		const indexByRun: Record<string, Map<number, number>> = {};
		for (const id of runIds) {
			const s = seriesByRun[id];
			if (!s) continue;
			const m = new Map<number, number>();
			s.steps.forEach((step, idx) => m.set(step, idx));
			indexByRun[id] = m;
		}

		const rows: ChartRow[] = sortedSteps.map((step) => {
			const row: ChartRow = { step };
			for (const id of runIds) {
				const s = seriesByRun[id];
				const idxMap = indexByRun[id];
				if (!s || !idxMap) {
					row[`smoothed:${id}`] = null;
					row[`raw:${id}`] = null;
					continue;
				}
				const i = idxMap.get(step);
				row[`smoothed:${id}`] = i !== undefined ? (s.smoothed[i] ?? null) : null;
				row[`raw:${id}`] = i !== undefined ? (s.raw[i] ?? null) : null;
			}
			return row;
		});

		if (rows.length <= MAX_RENDER_POINTS) return rows;

		// Pick which series to downsample against — use the run with the
		// most steps so we keep the densest line's shape. Then take rows
		// at the chosen step values; the other series stay aligned.
		let densestRun: string | null = null;
		let densestLen = 0;
		for (const id of runIds) {
			const len = seriesByRun[id]?.steps.length ?? 0;
			if (len > densestLen) {
				densestLen = len;
				densestRun = id;
			}
		}
		if (densestRun === null) return rows;
		const driver = lttb(
			rows.map((r) => ({
				x: r.step,
				y: (r[`smoothed:${densestRun}`] as number | null) ?? 0,
			})),
			MAX_RENDER_POINTS,
		);
		const keepSteps = new Set(driver.map((p) => p.x));
		return rows.filter((r) => keepSteps.has(r.step));
	}, [runIds, seriesByRun]);

	if (isLoading && data.length === 0) {
		return <Skeleton className="h-[320px] w-full" />;
	}
	if (data.length === 0) {
		return (
			<div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
				No loss frames available for the selected run
				{runIds.length === 1 ? "" : "s"} yet.
			</div>
		);
	}

	const lineDefs = runIds.flatMap((id, i) => {
		if (!seriesByRun[id]) return [];
		const color = colorForIndex(i);
		return [
			<Line
				key={`raw-${id}`}
				type="monotone"
				dataKey={`raw:${id}`}
				stroke={color}
				strokeOpacity={0.18}
				strokeWidth={1}
				dot={false}
				isAnimationActive={false}
				connectNulls
				name={`${shortRunLabel(id)} (raw)`}
			/>,
			<Line
				key={`smoothed-${id}`}
				type="monotone"
				dataKey={`smoothed:${id}`}
				stroke={color}
				strokeWidth={2}
				dot={false}
				isAnimationActive={false}
				connectNulls
				name={shortRunLabel(id)}
			/>,
		];
	});

	const renderedRuns = runIds.filter((id) => seriesByRun[id]);

	return (
		<div className="flex flex-col gap-3">
			<div className="h-[320px] w-full">
				<ResponsiveContainer width="100%" height="100%">
					<LineChart
						data={data}
						margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
					>
						<CartesianGrid
							stroke="var(--grid)"
							strokeDasharray="3 3"
							vertical={false}
						/>
						<XAxis
							dataKey="step"
							type="number"
							domain={["dataMin", "dataMax"]}
							tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
							stroke="var(--border)"
							tickFormatter={(v: number) => v.toLocaleString()}
						/>
						<YAxis
							tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
							stroke="var(--border)"
							domain={["auto", "auto"]}
							width={48}
							tickFormatter={(v: number) => v.toFixed(3)}
						/>
						<Tooltip
							cursor={{ stroke: "var(--accent-bracket)", strokeWidth: 1 }}
							contentStyle={{
								background: "var(--popover)",
								border: "1px solid var(--border)",
								borderRadius: "8px",
								fontSize: "12px",
							}}
							labelFormatter={(label) =>
								`step ${(label as number).toLocaleString()}`
							}
							formatter={(value, name) => [
								typeof value === "number" ? value.toFixed(5) : String(value),
								String(name),
							]}
						/>
						{lineDefs}
					</LineChart>
				</ResponsiveContainer>
			</div>
			<LossChartLegend runIds={renderedRuns} />
		</div>
	);
}

function shortRunLabel(runId: string): string {
	// Mirror shortenRunId(runId, 26) so legend stays readable.
	if (runId.length <= 26) return runId;
	return `${runId.slice(0, 12)}…${runId.slice(-8)}`;
}
