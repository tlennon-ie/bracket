// Side-by-side comparison: pick up to 3 runs, line up samples by prompt index.

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { shortenRunId } from "@/lib/format";
import type { GalleryGroup, GalleryItem } from "@/types/domain";
import { useMemo } from "react";

interface Props {
	groups: GalleryGroup[];
	runIds: string[];
}

interface PromptRow {
	id: string;
	promptIndex: number;
	cells: { runId: string; item: GalleryItem | undefined }[];
}

export function ComparisonGrid({ groups, runIds }: Props) {
	const rows = useMemo<PromptRow[]>(() => {
		const byId = new Map(groups.map((g) => [g.run_id, g]));
		const indexed = runIds.map((id) => {
			const grp = byId.get(id);
			return [...(grp?.items ?? [])].sort((a, b) =>
				a.caption.localeCompare(b.caption),
			);
		});
		const maxLen = Math.max(0, ...indexed.map((arr) => arr.length));
		const out: PromptRow[] = [];
		for (let p = 0; p < maxLen; p++) {
			const cells = runIds.map((id, runIdx) => ({
				runId: id,
				item: indexed[runIdx]?.[p],
			}));
			// Stable id based on the captions of the cells — survives reorder.
			const captionKey = cells.map((c) => c.item?.caption ?? "_").join("|");
			out.push({
				id: `${p}-${captionKey}`,
				promptIndex: p,
				cells,
			});
		}
		return out;
	}, [groups, runIds]);

	if (runIds.length === 0) {
		return (
			<p className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg">
				Tick up to 3 runs in the ledger to compare.
			</p>
		);
	}

	if (rows.length === 0) {
		return (
			<p className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg">
				Selected runs have no samples.
			</p>
		);
	}

	return (
		<div className="flex flex-col gap-6">
			{rows.map((row) => (
				<Card key={row.id}>
					<CardHeader>
						<CardTitle className="text-sm font-medium">
							Prompt #{row.promptIndex}
						</CardTitle>
					</CardHeader>
					<CardContent>
						<div
							className="grid gap-3"
							style={{
								gridTemplateColumns: `repeat(${runIds.length}, minmax(0, 1fr))`,
							}}
						>
							{row.cells.map((cell) => (
								<div
									key={`${row.id}-${cell.runId}`}
									className="flex flex-col gap-1.5"
								>
									<div className="text-[10px] uppercase tracking-wider text-muted-foreground">
										<code className="font-mono-tight">
											{shortenRunId(cell.runId, 28)}
										</code>
									</div>
									{cell.item ? (
										<img
											loading="lazy"
											src={cell.item.url}
											alt={cell.item.caption}
											className="w-full h-auto rounded-md border border-border"
										/>
									) : (
										<div className="aspect-square w-full rounded-md bg-muted flex items-center justify-center text-xs text-muted-foreground">
											no sample
										</div>
									)}
									{cell.item && (
										<code className="text-[10px] text-muted-foreground font-mono-tight truncate">
											{cell.item.caption}
										</code>
									)}
								</div>
							))}
						</div>
					</CardContent>
				</Card>
			))}
		</div>
	);
}
