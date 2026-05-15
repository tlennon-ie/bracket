// ConsoleLog — live tail of a run's stdout.log on the Monitor page.
//
// Polls /api/runs/<id>/log every 1.5s via useRunLog and renders the tail
// inside a fixed-height monospace viewport. Auto-scrolls to the bottom
// while the user is already at the bottom; if they scroll up to inspect
// earlier output, auto-scroll pauses and a "Resume tail" button appears.

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useRunLog } from "@/hooks/useRunLog";
import { ArrowDownToLine } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

interface Props {
	runId: string | null | undefined;
}

// Treat "within this many px from the bottom" as still-at-bottom — gives
// a little slack so scrollbar quirks don't disable auto-scroll.
const BOTTOM_EPSILON = 16;

export function ConsoleLog({ runId }: Props) {
	const { text, totalSize, exists, error, isLive } = useRunLog({ runId });
	const viewportRef = useRef<HTMLPreElement | null>(null);
	const [stickToBottom, setStickToBottom] = useState(true);

	// Auto-scroll on new bytes only when the user hasn't scrolled away.
	useLayoutEffect(() => {
		const el = viewportRef.current;
		if (!el || !stickToBottom) return;
		el.scrollTop = el.scrollHeight;
	}, [text, stickToBottom]);

	// Reset stickiness when the run changes — the new file starts at the
	// bottom by default.
	useEffect(() => {
		setStickToBottom(true);
	}, [runId]);

	function onScroll(): void {
		const el = viewportRef.current;
		if (!el) return;
		const distanceFromBottom =
			el.scrollHeight - el.clientHeight - el.scrollTop;
		setStickToBottom(distanceFromBottom <= BOTTOM_EPSILON);
	}

	function jumpToBottom(): void {
		const el = viewportRef.current;
		if (!el) return;
		el.scrollTop = el.scrollHeight;
		setStickToBottom(true);
	}

	if (!runId) {
		return (
			<p className="text-xs text-muted-foreground">
				No run selected — start a session to see live trainer output.
			</p>
		);
	}

	return (
		<div className="relative flex flex-col gap-2">
			<div className="flex items-center justify-between text-xs text-muted-foreground">
				<span className="font-mono-tight">
					stdout.log
					{exists && (
						<>
							{" · "}
							{formatBytes(totalSize)}
						</>
					)}
				</span>
				<div className="flex items-center gap-2">
					{error && (
						<span className="text-destructive font-mono-tight">{error}</span>
					)}
					{!stickToBottom && (
						<Button
							size="sm"
							variant="outline"
							className="h-6 px-2 text-xs"
							onClick={jumpToBottom}
						>
							<ArrowDownToLine className="h-3 w-3 mr-1" />
							Tail
						</Button>
					)}
					<span
						className={
							isLive
								? "text-success font-mono-tight"
								: "text-muted-foreground font-mono-tight"
						}
					>
						{isLive ? "live" : "waiting"}
					</span>
				</div>
			</div>
			{!exists ? (
				<Skeleton className="h-72 w-full" />
			) : (
				<pre
					ref={viewportRef}
					onScroll={onScroll}
					className="h-72 overflow-auto rounded-md border bg-zinc-950 text-zinc-100 text-xs leading-relaxed p-3 font-mono-tight whitespace-pre"
				>
					{text || " "}
				</pre>
			)}
		</div>
	);
}

function formatBytes(n: number): string {
	if (n < 1024) return `${n} B`;
	if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
	return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
