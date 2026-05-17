// Compact legend rendered below RunComparisonChart. Recharts' built-in
// legend doesn't have a good story for "one logical entry per run with
// two stroked lines (raw + smoothed)" — easier to render our own.

import { colorForIndex } from "@/components/results/RunComparisonChart";
import { shortenRunId } from "@/lib/format";

interface Props {
	runIds: string[];
}

export function LossChartLegend({ runIds }: Props) {
	if (runIds.length === 0) return null;
	return (
		<ul className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs font-mono-tight">
			{runIds.map((id, i) => (
				<li key={id} className="flex items-center gap-2">
					<span
						className="inline-block h-0.5 w-6 rounded-full"
						style={{ background: colorForIndex(i) }}
						aria-hidden="true"
					/>
					<span className="text-foreground">{shortenRunId(id, 26)}</span>
				</li>
			))}
		</ul>
	);
}
