// useRunLog — tails a run's stdout.log via byte-offset polling.
//
// Single source of truth for the live console pane on the Monitor page.
// REST polling (not WebSocket): the trainer writes lots of bytes, the
// existing snapshot WS already does the per-second cadence, and polling
// is trivially cancellable / pausable per-run.

import { api } from "@/lib/api";
import type { RunLogChunk } from "@/types/domain";
import { useEffect, useRef, useState } from "react";

const POLL_INTERVAL_MS = 1500;
// Soft cap on in-memory text. The viewer drops the oldest bytes when this
// is exceeded so a 5-hour run doesn't grow an unbounded DOM string.
const MAX_BUFFER_BYTES = 512 * 1024;

interface UseRunLogResult {
	text: string;
	totalSize: number;
	isLive: boolean;
	exists: boolean;
	error: string | null;
	/** Drop the accumulated buffer and start fetching from the file head. */
	reset: () => void;
}

interface UseRunLogOptions {
	runId: string | null | undefined;
	enabled?: boolean;
}

export function useRunLog({
	runId,
	enabled = true,
}: UseRunLogOptions): UseRunLogResult {
	const [text, setText] = useState("");
	const [totalSize, setTotalSize] = useState(0);
	const [exists, setExists] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [isLive, setIsLive] = useState(false);

	// Refs hold mutable polling state without re-triggering effects.
	const offsetRef = useRef(0);
	const timerRef = useRef<number | null>(null);
	const cancelledRef = useRef(false);

	useEffect(() => {
		cancelledRef.current = false;
		// Reset accumulated state whenever the target run changes.
		setText("");
		setTotalSize(0);
		setExists(false);
		setError(null);
		setIsLive(false);
		offsetRef.current = 0;

		if (!runId || !enabled) {
			return () => {
				cancelledRef.current = true;
				if (timerRef.current !== null) {
					window.clearTimeout(timerRef.current);
					timerRef.current = null;
				}
			};
		}

		async function poll(): Promise<void> {
			if (cancelledRef.current || !runId) return;
			try {
				const chunk: RunLogChunk = await api.getRunLog(
					runId,
					offsetRef.current,
				);
				if (cancelledRef.current) return;
				setError(null);
				setExists(chunk.exists);
				setTotalSize(chunk.total_size);
				if (chunk.exists) {
					// If the server tailed past our requested offset (truncation
					// or our offset went stale), reset the displayed text so the
					// chunk we receive becomes the new ground truth.
					if (chunk.truncated_to_tail || chunk.offset < offsetRef.current) {
						setText(chunk.content);
					} else if (chunk.content.length > 0) {
						setText((prev) => trimToBuffer(prev + chunk.content));
					}
					offsetRef.current = chunk.next_offset;
					setIsLive(true);
				}
			} catch (e) {
				if (!cancelledRef.current) {
					setError(e instanceof Error ? e.message : "log fetch failed");
				}
			} finally {
				if (!cancelledRef.current) {
					timerRef.current = window.setTimeout(poll, POLL_INTERVAL_MS);
				}
			}
		}

		void poll();

		return () => {
			cancelledRef.current = true;
			if (timerRef.current !== null) {
				window.clearTimeout(timerRef.current);
				timerRef.current = null;
			}
		};
	}, [runId, enabled]);

	function reset(): void {
		offsetRef.current = 0;
		setText("");
	}

	return { text, totalSize, isLive, exists, error, reset };
}

function trimToBuffer(s: string): string {
	if (s.length <= MAX_BUFFER_BYTES) return s;
	// Drop oldest bytes, then snap to the next newline so we don't start
	// rendering mid-line and confuse the user.
	const dropTo = s.length - MAX_BUFFER_BYTES;
	const nl = s.indexOf("\n", dropTo);
	return nl >= 0 ? s.slice(nl + 1) : s.slice(dropTo);
}
