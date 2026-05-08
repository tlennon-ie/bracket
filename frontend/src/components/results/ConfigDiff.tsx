// Two-column config diff. Pure-render, no editor.
//
// Each row shows the key, the value from runs[0], runs[1], runs[2]; values
// that differ from the leftmost are highlighted.

import { cn } from "@/lib/utils";
import type { RunDetail } from "@/types/domain";
import { useMemo } from "react";

interface Props {
	runs: RunDetail[];
}

export function ConfigDiff({ runs }: Props) {
	const allKeys = useMemo(() => {
		const set = new Set<string>();
		for (const r of runs) for (const k of Object.keys(r.config)) set.add(k);
		return [...set].sort();
	}, [runs]);

	if (runs.length < 2) {
		return (
			<p className="text-xs text-muted-foreground">
				Pick at least 2 runs to diff.
			</p>
		);
	}

	return (
		<div className="overflow-x-auto">
			<table className="w-full text-xs border-collapse">
				<thead>
					<tr className="text-[10px] text-muted-foreground uppercase tracking-wider border-b border-border">
						<th className="text-left font-medium py-2 pr-3 sticky left-0 bg-card">
							Key
						</th>
						{runs.map((r) => (
							<th
								key={r.run_id}
								className="text-left font-medium py-2 pr-3 font-mono-tight"
							>
								{r.run_id}
							</th>
						))}
					</tr>
				</thead>
				<tbody>
					{allKeys.map((k) => {
						const ref = runs[0]?.config[k];
						return (
							<tr key={k} className="border-b border-border/60">
								<td className="py-1.5 pr-3 font-mono-tight text-muted-foreground sticky left-0 bg-card">
									{k}
								</td>
								{runs.map((r) => {
									const v = r.config[k] ?? "—";
									const differs = v !== ref;
									return (
										<td
											key={r.run_id}
											className={cn(
												"py-1.5 pr-3 font-mono-tight",
												differs && "text-warning bg-warning/5",
											)}
										>
											{v}
										</td>
									);
								})}
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
}
