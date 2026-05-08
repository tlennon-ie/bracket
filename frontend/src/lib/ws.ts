// Minimal reconnecting WebSocket. ~80 LOC, no extra deps.
//
// Backoff: 1s → 2s → 4s → 8s → 16s → 30s (cap).
// Heartbeat: every 30s send a ping; if no message for >45s, force-reconnect.
// Server is at /api/ws/snapshot (see bracket/api/server.py).

export type WSStatus = "connecting" | "open" | "closed" | "error";

export interface WSOptions<T> {
	url: string;
	onMessage: (msg: T) => void;
	onStatus?: (s: WSStatus) => void;
	/** Called on every successful (re)connect — use to resync via REST. */
	onOpen?: () => void;
	/** Optional message-validator. Throw to drop a malformed frame. */
	parse?: (raw: unknown) => T;
}

export interface WSHandle {
	close(): void;
	send(payload: unknown): void;
	status(): WSStatus;
}

const BACKOFF_LADDER_MS = [1000, 2000, 4000, 8000, 16000, 30000];

export function createReconnectingWS<T>(opts: WSOptions<T>): WSHandle {
	let socket: WebSocket | null = null;
	let backoffIdx = 0;
	let closed = false;
	let lastMsgAt = Date.now();
	let heartbeatTimer: ReturnType<typeof setInterval> | null = null;
	let watchdogTimer: ReturnType<typeof setInterval> | null = null;
	let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	let currentStatus: WSStatus = "connecting";

	const setStatus = (s: WSStatus) => {
		currentStatus = s;
		opts.onStatus?.(s);
	};

	const buildAbsoluteWsUrl = (rel: string): string => {
		if (rel.startsWith("ws://") || rel.startsWith("wss://")) return rel;
		if (typeof window === "undefined") return rel;
		const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
		return `${proto}//${window.location.host}${rel.startsWith("/") ? rel : `/${rel}`}`;
	};

	const stopTimers = () => {
		if (heartbeatTimer) clearInterval(heartbeatTimer);
		if (watchdogTimer) clearInterval(watchdogTimer);
		heartbeatTimer = null;
		watchdogTimer = null;
	};

	const connect = () => {
		if (closed) return;
		setStatus("connecting");
		try {
			socket = new WebSocket(buildAbsoluteWsUrl(opts.url));
		} catch {
			scheduleReconnect();
			return;
		}

		socket.onopen = () => {
			backoffIdx = 0;
			lastMsgAt = Date.now();
			setStatus("open");
			opts.onOpen?.();
			// 30s heartbeat — server doesn't strictly require it, keeps proxies alive.
			heartbeatTimer = setInterval(() => {
				try {
					socket?.send(JSON.stringify({ type: "ping" }));
				} catch {
					// ignore
				}
			}, 30_000);
			// Watchdog: if 45s of silence, kick the socket.
			watchdogTimer = setInterval(() => {
				if (Date.now() - lastMsgAt > 45_000) {
					try {
						socket?.close();
					} catch {
						// ignore
					}
				}
			}, 5_000);
		};

		socket.onmessage = (ev: MessageEvent) => {
			lastMsgAt = Date.now();
			try {
				const data = JSON.parse(ev.data as string);
				const msg = opts.parse ? opts.parse(data) : (data as T);
				opts.onMessage(msg);
			} catch {
				// Drop malformed frame; never throw out of an event handler.
			}
		};

		socket.onerror = () => {
			setStatus("error");
		};

		socket.onclose = () => {
			stopTimers();
			socket = null;
			if (closed) return;
			setStatus("closed");
			scheduleReconnect();
		};
	};

	const scheduleReconnect = () => {
		if (closed) return;
		const delay =
			BACKOFF_LADDER_MS[Math.min(backoffIdx, BACKOFF_LADDER_MS.length - 1)] ??
			30_000;
		backoffIdx++;
		reconnectTimer = setTimeout(connect, delay);
	};

	connect();

	return {
		close() {
			closed = true;
			stopTimers();
			if (reconnectTimer) clearTimeout(reconnectTimer);
			try {
				socket?.close();
			} catch {
				// ignore
			}
		},
		send(payload: unknown) {
			try {
				socket?.send(
					typeof payload === "string" ? payload : JSON.stringify(payload),
				);
			} catch {
				// ignore
			}
		},
		status: () => currentStatus,
	};
}
