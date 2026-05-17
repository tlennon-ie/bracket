// LiveLossChart — Recharts LineChart with raw + client-side smoothed series.
// Smoothing slider value comes from the UI store; raw points come from WS.

import { Skeleton } from "@/components/ui/skeleton";
import { applyEMA } from "@/lib/ema";
import { lttb } from "@/lib/lttb";
import { useUIStore } from "@/store/ui";
import type { LossSeries } from "@/types/domain";
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
	series: LossSeries | null | undefined;
	isLoading?: boolean;
}

const MAX_RENDER_POINTS = 1000;

export function LossChart({ series, isLoading }: Props) {
	const smoothing = useUIStore((s) => s.smoothing);

	const data = useMemo(() => {
		if (!series || series.steps.length === 0) return [];
		const smoothed = applyEMA(series.raw, smoothing);
		const gradNorms = series.grad_norms ?? [];
		const points = series.steps.map((step, i) => ({
			step,
			raw: series.raw[i] ?? null,
			smoothed: smoothed[i] ?? null,
			grad_norm: gradNorms[i] ?? null,
		}));
		if (points.length <= MAX_RENDER_POINTS) return points;
		// Downsample raw + smoothed independently for faithful LTTB.
		const rawDS = lttb(
			points.map((p) => ({ x: p.step, y: p.raw ?? 0 })),
			MAX_RENDER_POINTS,
		);
		// Build a step → grad_norm map so we can carry the secondary
		// series through downsampling without running LTTB twice.
		const gradByStep = new Map<number, number | null>();
		for (const p of points) gradByStep.set(p.step, p.grad_norm);
		return rawDS.map((p, i) => ({
			step: p.x,
			raw: p.y,
			smoothed:
				applyEMA(
					rawDS.slice(0, i + 1).map((q) => q.y),
					smoothing,
				).pop() ?? null,
			grad_norm: gradByStep.get(p.x) ?? null,
		}));
	}, [series, smoothing]);

	const hasGradNorm = useMemo(
		() => data.some((d) => typeof d.grad_norm === "number"),
		[data],
	);

	if (isLoading) {
		return <Skeleton className="h-[260px] w-full" />;
	}
	if (data.length === 0) {
		return (
			<div className="flex h-[260px] items-center justify-center text-sm text-muted-foreground">
				Waiting for the first loss frame…
			</div>
		);
	}

	return (
		<div className="h-[260px] w-full">
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
						yAxisId="left"
						tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
						stroke="var(--border)"
						domain={["auto", "auto"]}
						width={48}
						tickFormatter={(v: number) => v.toFixed(3)}
					/>
					{hasGradNorm ? (
						<YAxis
							yAxisId="right"
							orientation="right"
							tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
							stroke="var(--border)"
							domain={["auto", "auto"]}
							width={48}
							tickFormatter={(v: number) => v.toFixed(2)}
						/>
					) : null}
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
							name === "raw" ? "raw" : "smoothed",
						]}
					/>
					<Line
						yAxisId="left"
						type="monotone"
						dataKey="raw"
						stroke="var(--muted-foreground)"
						strokeWidth={1}
						dot={false}
						isAnimationActive={false}
						name="raw"
					/>
					<Line
						yAxisId="left"
						type="monotone"
						dataKey="smoothed"
						stroke="var(--accent-bracket)"
						strokeWidth={2}
						dot={false}
						isAnimationActive={false}
						name="smoothed"
					/>
					{hasGradNorm ? (
						<Line
							yAxisId="right"
							type="monotone"
							dataKey="grad_norm"
							stroke="var(--muted-foreground)"
							strokeWidth={1}
							strokeOpacity={0.5}
							strokeDasharray="4 4"
							dot={false}
							isAnimationActive={false}
							name="grad_norm"
							connectNulls
						/>
					) : null}
				</LineChart>
			</ResponsiveContainer>
		</div>
	);
}
