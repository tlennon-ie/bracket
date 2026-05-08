import { GalleryAccordion } from "@/components/monitor/GalleryAccordion";
import { LossChart } from "@/components/monitor/LossChart";
import { ProgressBar } from "@/components/monitor/ProgressBar";
import { ScoreHistoryTable } from "@/components/monitor/ScoreHistoryTable";
import { SmoothingSlider } from "@/components/monitor/SmoothingSlider";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useGallery } from "@/hooks/useGallery";
import { useSnapshot } from "@/hooks/useSnapshot";
import { formatDuration } from "@/lib/format";
import { Link, createFileRoute } from "@tanstack/react-router";
import { Activity, AlertCircle } from "lucide-react";

export const Route = createFileRoute("/monitor")({
	component: MonitorPage,
});

function MonitorPage() {
	const { snapshot, isLoading, wsStatus } = useSnapshot();
	const gallery = useGallery();

	// ── Empty state ────────────────────────────────────────────────────────────
	if (!isLoading && (!snapshot || snapshot.session_status === "idle")) {
		return (
			<div className="flex flex-col items-center justify-center py-24 text-center gap-4">
				<Activity className="h-10 w-10 text-muted-foreground" />
				<div className="flex flex-col gap-1">
					<h2 className="text-xl font-semibold tracking-tight">
						No session yet
					</h2>
					<p className="text-sm text-muted-foreground max-w-md">
						Start an orchestration run from <strong>Setup</strong>, then live
						progress shows up here.
					</p>
				</div>
				<Button asChild variant="default">
					<Link to="/setup">Start a session →</Link>
				</Button>
			</div>
		);
	}

	return (
		<div className="flex flex-col gap-6">
			<header className="flex flex-wrap items-end justify-between gap-3">
				<div className="flex flex-col gap-1">
					<h1 className="text-3xl font-semibold tracking-tight font-display">
						Monitor
					</h1>
					<p className="text-sm text-muted-foreground">
						Live status. Loss chart smooths client-side — drag the slider; no
						roundtrip.
					</p>
				</div>
				<div className="flex items-center gap-2 text-xs">
					<Badge
						variant={
							wsStatus === "open"
								? "success"
								: wsStatus === "connecting"
									? "warning"
									: "destructive"
						}
						className="uppercase"
					>
						ws · {wsStatus}
					</Badge>
				</div>
			</header>

			{/* ── Status header ───────────────────────────────────────────────── */}
			<Card>
				<CardContent className="flex flex-col gap-4 pt-6">
					<div className="flex flex-wrap items-center gap-3">
						{snapshot ? (
							<>
								<Badge
									variant={
										snapshot.session_status === "running"
											? "success"
											: snapshot.session_status === "error"
												? "destructive"
												: snapshot.session_status === "done"
													? "default"
													: "muted"
									}
									className="uppercase tracking-wider"
								>
									{snapshot.session_status}
								</Badge>
								{snapshot.current_run_id && (
									<code className="font-mono-tight text-sm">
										{snapshot.current_run_id}
									</code>
								)}
								{snapshot.current_run_steps_done !== null &&
									snapshot.current_run_max_steps !== null && (
										<span className="text-xs text-muted-foreground font-mono-tight">
											step {snapshot.current_run_steps_done}/
											{snapshot.current_run_max_steps}
										</span>
									)}
								<span className="text-xs text-muted-foreground font-mono-tight">
									elapsed {formatDuration(snapshot.elapsed_s)}
								</span>
							</>
						) : (
							<Skeleton className="h-5 w-64" />
						)}
					</div>

					{snapshot && (
						<ProgressBar
							pct={snapshot.progress_pct}
							completed={snapshot.completed_runs}
							total={snapshot.total_runs_target}
						/>
					)}

					{snapshot?.judge_summary && (
						<p className="text-xs text-muted-foreground">
							{snapshot.judge_summary}
						</p>
					)}

					{snapshot?.error_message && (
						<div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs">
							<AlertCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
							<pre className="whitespace-pre-wrap font-mono-tight text-destructive">
								{snapshot.error_message}
							</pre>
						</div>
					)}
				</CardContent>
			</Card>

			{/* ── Loss chart ──────────────────────────────────────────────────── */}
			<Card>
				<CardHeader className="flex-row items-center justify-between space-y-0">
					<div className="flex flex-col gap-0.5">
						<CardTitle>Live loss</CardTitle>
						{snapshot?.current_run_id && (
							<code className="text-xs font-mono-tight text-muted-foreground">
								{snapshot.current_run_id}
							</code>
						)}
					</div>
					<SmoothingSlider />
				</CardHeader>
				<CardContent>
					<LossChart
						series={snapshot?.current_loss ?? null}
						isLoading={isLoading}
					/>
				</CardContent>
			</Card>

			{/* ── Score history ───────────────────────────────────────────────── */}
			<Card>
				<CardHeader>
					<CardTitle>Score history</CardTitle>
				</CardHeader>
				<CardContent>
					<ScoreHistoryTable rows={snapshot?.score_history ?? []} />
				</CardContent>
			</Card>

			{/* ── Gallery ─────────────────────────────────────────────────────── */}
			<Card>
				<CardHeader>
					<CardTitle>Samples</CardTitle>
				</CardHeader>
				<CardContent>
					{gallery.isLoading ? (
						<Skeleton className="h-32 w-full" />
					) : (
						<GalleryAccordion groups={gallery.data ?? []} />
					)}
				</CardContent>
			</Card>
		</div>
	);
}
