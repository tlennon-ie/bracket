// useSnapshot — bridges TanStack Query + WebSocket.
//
// REST is authoritative on initial mount and reconnect.
// WS pushes deltas; we mirror them into the query cache.

import { api } from "@/lib/api";
import { type WSStatus, createReconnectingWS } from "@/lib/ws";
import type { MonitorSnapshot } from "@/types/domain";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

const QUERY_KEY = ["snapshot"] as const;

export function useSnapshot(): {
	snapshot: MonitorSnapshot | undefined;
	isLoading: boolean;
	wsStatus: WSStatus;
	error: unknown;
} {
	const qc = useQueryClient();
	const [wsStatus, setWsStatus] = useState<WSStatus>("connecting");
	const wsHandle = useRef<ReturnType<
		typeof createReconnectingWS<MonitorSnapshot>
	> | null>(null);

	const query = useQuery({
		queryKey: QUERY_KEY,
		queryFn: () => api.getSnapshot(true),
		staleTime: 0,
		refetchOnWindowFocus: true,
		retry: (count) => count < 3,
	});

	useEffect(() => {
		if (wsHandle.current) return;
		const handle = createReconnectingWS<MonitorSnapshot>({
			url: "/api/ws/snapshot",
			onMessage: (msg) => {
				qc.setQueryData(QUERY_KEY, msg);
			},
			onStatus: setWsStatus,
			onOpen: () => {
				// Resync with REST on reconnect — server is the source of truth.
				void qc.invalidateQueries({ queryKey: QUERY_KEY });
			},
		});
		wsHandle.current = handle;
		return () => {
			handle.close();
			wsHandle.current = null;
		};
	}, [qc]);

	return {
		snapshot: query.data,
		isLoading: query.isLoading,
		wsStatus,
		error: query.error,
	};
}
