import { Sparkline } from "@/components/monitor/Sparkline";
import { Badge } from "@/components/ui/badge";
import { formatDuration, formatScore, shortenRunId } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { CandidateRow } from "@/types/domain";
import { Link } from "@tanstack/react-router";
import { Trophy } from "lucide-react";
import { useMemo } from "react";

interface Props {
	rows: CandidateRow[];
	/** Per-runId loss series (raw values) for sparkline column. */
	sparkSeries?: Record<string, number[]>;
}

export function ScoreHistoryTable({ rows, sparkSeries }: Props) {
	const bestRunId = useMemo(() => {
		let best: CandidateRow | null = null;
		for (const r of rows) {
			if (r.score === null || r.disqualified) continue;
			if (!best || (best.score ?? Number.POSITIVE_INFINITY) > r.score) best = r;
		}
		return best?.run_id ?? null;
	}, [rows]);

	if (rows.length === 0) {
		return (
			<div className="text-sm text-muted-foreground py-8 text-center border border-dashed border-border rounded-lg">
				No runs scored yet.
			</div>
		);
	}

	return (
		<div className="overflow-x-auto -mx-1 px-1">
			<table className="w-full text-sm border-collapse">
				<thead>
					<tr className="text-[10px] text-muted-foreground uppercase tracking-wider border-b border-border">
						<th className="text-left font-medium py-2 pr-3">Run</th>
						<th className="text-left font-medium py-2 pr-3">Role</th>
						<th className="text-left font-medium py-2 pr-3">Config</th>
						<th className="text-right font-medium py-2 pr-3">Score</th>
						<th className="text-right font-medium py-2 pr-3">Final</th>
						<th className="text-right font-medium py-2 pr-3">Slope</th>
						<th className="text-right font-medium py-2 pr-3">Steps</th>
						<th className="text-right font-medium py-2 pr-3">Duration</th>
						<th className="text-left font-medium py-2 pr-3">Trajectory</th>
					</tr>
				</thead>
				<tbody>
					{rows.map((r) => {
						const isBest = r.run_id === bestRunId;
						const dq = !!r.disqualified;
						const series = sparkSeries?.[r.run_id];
						return (
							<tr
								key={r.run_id}
								className={cn(
									"border-b border-border/60 hover:bg-accent/40 transition-colors",
									isBest && "bg-primary/5",
								)}
							>
								<td className="py-2 pr-3">
									<Link
										to="/results"
										search={{ run: r.run_id }}
										className="flex items-center gap-1.5 text-foreground hover:underline"
									>
										{isBest && <Trophy className="h-3.5 w-3.5 text-primary" />}
										<code className="font-mono-tight text-xs">
											{shortenRunId(r.run_id, 22)}
										</code>
									</Link>
								</td>
								<td className="py-2 pr-3">
									<Badge variant="outline" className="capitalize text-[10px]">
										{r.role}
									</Badge>
								</td>
								<td className="py-2 pr-3">
									<code className="font-mono-tight text-xs text-muted-foreground">
										{shortenRunId(r.config_id, 14)}
									</code>
								</td>
								<td className="py-2 pr-3 text-right font-mono-tight tabular-nums">
									{dq ? (
										<span className="text-destructive text-xs">DQ</span>
									) : (
										formatScore(r.score)
									)}
								</td>
								<td className="py-2 pr-3 text-right font-mono-tight tabular-nums text-muted-foreground">
									{formatScore(r.final_smoothed)}
								</td>
								<td className="py-2 pr-3 text-right font-mono-tight tabular-nums text-muted-foreground">
									{formatScore(r.slope, 4)}
								</td>
								<td className="py-2 pr-3 text-right font-mono-tight tabular-nums text-muted-foreground">
									{r.n_steps.toLocaleString()}
								</td>
								<td className="py-2 pr-3 text-right font-mono-tight tabular-nums text-muted-foreground">
									{formatDuration(r.duration_s)}
								</td>
								<td className="py-2 pr-3">
									{series && series.length > 1 ? (
										<Sparkline values={series} />
									) : (
										<span className="text-muted-foreground text-xs">—</span>
									)}
								</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
}
