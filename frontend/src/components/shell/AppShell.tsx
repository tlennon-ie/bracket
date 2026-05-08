import { NavRail } from "@/components/shell/NavRail";
import { RunControlsBar } from "@/components/shell/RunControlsBar";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcut";
import { useTheme } from "@/hooks/useTheme";
import { api } from "@/lib/api";
import { useUIStore } from "@/store/ui";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { Toaster } from "sonner";

interface AppShellProps {
	children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
	const { resolved } = useTheme();
	const qc = useQueryClient();
	const navigate = useNavigate();
	const setSmoothing = useUIStore((s) => s.setSmoothing);
	const smoothing = useUIStore((s) => s.smoothing);

	const stopMutation = useMutation({
		mutationFn: api.stopSession,
		onSuccess: () => qc.invalidateQueries({ queryKey: ["snapshot"] }),
	});

	// Keyboard shortcuts (per migration plan §7).
	useKeyboardShortcuts([
		{ key: "r", handler: () => qc.invalidateQueries() },
		{ key: "Escape", handler: () => stopMutation.mutate() },
		{ chord: "g", key: "s", handler: () => navigate({ to: "/setup" }) },
		{ chord: "g", key: "r", handler: () => navigate({ to: "/run" }) },
		{ chord: "g", key: "m", handler: () => navigate({ to: "/monitor" }) },
		{ chord: "g", key: "o", handler: () => navigate({ to: "/results" }) },
		{
			key: "[",
			handler: () => setSmoothing(Math.max(0, smoothing - 0.05)),
		},
		{
			key: "]",
			handler: () => setSmoothing(Math.min(0.99, smoothing + 0.05)),
		},
	]);

	return (
		<div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
			<a href="#main" className="skip-link">
				Skip to content
			</a>
			<NavRail />
			<div className="flex flex-col flex-1 min-w-0">
				<main
					id="main"
					tabIndex={-1}
					className="flex-1 min-h-0 overflow-y-auto"
					aria-label="Main content"
				>
					<div className="mx-auto max-w-[1280px] px-6 py-8">{children}</div>
				</main>
				<RunControlsBar />
			</div>
			<Toaster
				theme={resolved}
				position="bottom-right"
				toastOptions={{ className: "font-sans" }}
				closeButton
			/>
		</div>
	);
}
