// Sticky bottom run-controls bar — visible from every route.
// Contains Stop button and the live status pill while a session is in flight.

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useSnapshot } from "@/hooks/useSnapshot";
import { api } from "@/lib/api";
import { formatDuration, shortenRunId } from "@/lib/format";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Square } from "lucide-react";
import { toast } from "sonner";

export function RunControlsBar() {
	const { snapshot } = useSnapshot();
	const qc = useQueryClient();

	const stopMutation = useMutation({
		mutationFn: api.stopSession,
		onSuccess: (res) => {
			if (res.stopped) {
				toast.success("Stop requested", { description: res.message });
			} else {
				toast.info("Nothing to stop", { description: res.message });
			}
			void qc.invalidateQueries({ queryKey: ["snapshot"] });
		},
	});

	if (!snapshot) return null;
	const {
		session_status,
		current_run_id,
		current_run_steps_done,
		current_run_max_steps,
		elapsed_s,
		progress_pct,
	} = snapshot;
	const isLive = session_status === "running" || session_status === "stopping";
	if (!isLive && session_status !== "done" && session_status !== "error")
		return null;

	return (
		<div
			data-run-controls
			className="border-t border-border bg-card/80 backdrop-blur supports-[backdrop-filter]:bg-card/60"
		>
			<div className="mx-auto flex max-w-[1280px] items-center gap-4 px-6 py-2.5">
				<Badge
					variant={
						session_status === "running"
							? "success"
							: session_status === "error"
								? "destructive"
								: session_status === "done"
									? "default"
									: "warning"
					}
					className="uppercase tracking-wider"
				>
					{session_status}
				</Badge>

				{current_run_id && (
					<div className="flex items-center gap-2 text-sm">
						<span className="text-muted-foreground text-xs uppercase tracking-wider">
							Run
						</span>
						<code className="font-mono-tight text-foreground">
							{shortenRunId(current_run_id)}
						</code>
						{current_run_steps_done !== null &&
							current_run_max_steps !== null && (
								<span className="text-muted-foreground font-mono-tight text-xs">
									step {current_run_steps_done}/{current_run_max_steps}
								</span>
							)}
					</div>
				)}

				<div className="flex items-center gap-2 text-xs text-muted-foreground font-mono-tight">
					<span>elapsed {formatDuration(elapsed_s)}</span>
					<span aria-hidden>·</span>
					<span>{progress_pct.toFixed(1)}%</span>
				</div>

				<div className="ml-auto flex items-center gap-2">
					<Button
						variant="destructive"
						size="sm"
						onClick={() => stopMutation.mutate()}
						disabled={!isLive || stopMutation.isPending}
						aria-label="Stop session (Esc)"
						className="gap-1.5"
					>
						<Square className="h-3.5 w-3.5 fill-current" />
						Stop
					</Button>
				</div>
			</div>
		</div>
	);
}
