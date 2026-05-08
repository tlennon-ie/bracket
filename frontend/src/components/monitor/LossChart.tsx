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
		const points = series.steps.map((step, i) => ({
			step,
			raw: series.raw[i] ?? null,
			smoothed: smoothed[i] ?? null,
		}));
		if (points.length <= MAX_RENDER_POINTS) return points;
		// Downsample raw + smoothed independently for faithful LTTB.
		const rawDS = lttb(
			points.map((p) => ({ x: p.step, y: p.raw ?? 0 })),
			MAX_RENDER_POINTS,
		);
		return rawDS.map((p, i) => ({
			step: p.x,
			raw: p.y,
			smoothed:
				applyEMA(
					rawDS.slice(0, i + 1).map((q) => q.y),
					smoothing,
				).pop() ?? null,
		}));
	}, [series, smoothing]);

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
							name === "raw" ? "raw" : "smoothed",
						]}
					/>
					<Line
						type="monotone"
						dataKey="raw"
						stroke="var(--muted-foreground)"
						strokeWidth={1}
						dot={false}
						isAnimationActive={false}
						name="raw"
					/>
					<Line
						type="monotone"
						dataKey="smoothed"
						stroke="var(--accent-bracket)"
						strokeWidth={2}
						dot={false}
						isAnimationActive={false}
						name="smoothed"
					/>
				</LineChart>
			</ResponsiveContainer>
		</div>
	);
}
