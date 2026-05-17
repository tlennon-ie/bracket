// ConsoleLog — live tail of a run's stdout.log on the Monitor page.
//
// Polls /api/runs/<id>/log every 1.5s via useRunLog and renders the tail
// inside a fixed-height monospace viewport. Auto-scrolls to the bottom
// while the user is already at the bottom; if they scroll up to inspect
// earlier output, auto-scroll pauses and a "Resume tail" button appears.
//
// tqdm progress bars emit ``\r``-terminated frames (one per step) instead
// of newlines so a terminal can overwrite the prior line. Browsers do
// nothing with ``\r`` inside ``<pre>`` — every frame stays glued to the
// previous one on a single line, producing the unreadable horizontal
// scroll the Monitor page had pre-fix. We collapse each ``\n``-bounded
// line to its last ``\r`` segment so each epoch occupies one row and the
// step counter updates in place.

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useRunLog } from "@/hooks/useRunLog";
import { ArrowDownToLine } from "lucide-react";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

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
	const displayText = useMemo(() => collapseCarriageReturns(text), [text]);

	// Auto-scroll on new bytes only when the user hasn't scrolled away.
	useLayoutEffect(() => {
		const el = viewportRef.current;
		if (!el || !stickToBottom) return;
		el.scrollTop = el.scrollHeight;
	}, [displayText, stickToBottom]);

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
					{displayText || " "}
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

// Browsers don't honour ``\r`` inside ``<pre>``; tqdm progress frames are
// ``\r``-delimited (each step is a frame). For each ``\n``-bounded line,
// keep only what follows the last ``\r`` — that's what a terminal would
// show after all the in-place overwrites resolved. Preserves leading
// ``\r`` on the first line of a chunk if the prior frame already had a
// final ``\n``.
export function collapseCarriageReturns(s: string): string {
	if (!s.includes("\r")) return s;
	const lines = s.split("\n");
	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		if (line === undefined || !line.includes("\r")) continue;
		const segments = line.split("\r");
		// The final segment is what's currently visible on that line.
		// If it's empty (line ends in ``\r``), fall back to the prior
		// non-empty segment so an in-progress frame still shows.
		let chosen: string = segments[segments.length - 1] ?? "";
		if (chosen.length === 0) {
			for (let j = segments.length - 2; j >= 0; j--) {
				const seg = segments[j];
				if (seg && seg.length > 0) {
					chosen = seg;
					break;
				}
			}
		}
		lines[i] = chosen;
	}
	return lines.join("\n");
}
