// Polls /api/update/check on mount and surfaces a sonner toast when a
// newer release is available. The user can dismiss (snoozed for the
// session via localStorage), open the release notes, or trigger
// /api/update/apply which spawns the platform updater script.

import { api } from "@/lib/api";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { toast } from "sonner";

const SNOOZE_STORAGE_KEY = "bracket.updateSnoozedVersion";
// Re-check every 6 hours while the tab stays open.
const REFETCH_INTERVAL_MS = 6 * 60 * 60 * 1000;

function readSnoozedVersion(): string | null {
	if (typeof window === "undefined") return null;
	try {
		return window.localStorage.getItem(SNOOZE_STORAGE_KEY);
	} catch {
		return null;
	}
}

function writeSnoozedVersion(version: string): void {
	if (typeof window === "undefined") return;
	try {
		window.localStorage.setItem(SNOOZE_STORAGE_KEY, version);
	} catch {
		// Storage may be unavailable (private browsing); fail silently —
		// the toast will reappear on next mount.
	}
}

export function useUpdateChecker(): void {
	const { data } = useQuery({
		queryKey: ["update-check"],
		queryFn: () => api.checkForUpdate(false),
		// Stale after 30 min — matches the backend cache TTL.
		staleTime: 30 * 60 * 1000,
		refetchInterval: REFETCH_INTERVAL_MS,
		refetchOnWindowFocus: false,
		refetchOnReconnect: false,
		retry: false,
	});

	const applyMutation = useMutation({
		mutationFn: api.applyUpdate,
		onSuccess: (res) => {
			toast.success("Updating Bracket…", {
				description:
					res.message ||
					"Pulling latest changes and rebuilding. The page will reconnect once the new server is up.",
				duration: 15000,
			});
		},
	});

	useEffect(() => {
		if (!data?.update_available || !data.latest_version) return;
		const snoozed = readSnoozedVersion();
		if (snoozed === data.latest_version) return;

		const toastId = `bracket-update-${data.latest_version}`;
		const description = `${data.current_version} → ${data.latest_version}. Pulls the latest commit, rebuilds, and re-launches.`;

		toast(`Bracket update available`, {
			id: toastId,
			description,
			duration: Number.POSITIVE_INFINITY,
			action: {
				label: "Update",
				onClick: () => {
					toast.dismiss(toastId);
					applyMutation.mutate();
				},
			},
			cancel: {
				label: "Dismiss",
				onClick: () => {
					if (data.latest_version) writeSnoozedVersion(data.latest_version);
					toast.dismiss(toastId);
				},
			},
		});
		// Cleanup: nothing to do — sonner persists toasts past unmount and
		// dismissals are explicit.
	}, [data, applyMutation]);
}
