import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import { estimateWallSeconds, formatDuration } from "@/lib/format";
import { useSessionConfigStore } from "@/store/sessionConfig";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Clock, Play } from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/run")({
	component: RunPage,
});

function RunPage() {
	const config = useSessionConfigStore((s) => s.config);
	const setField = useSessionConfigStore((s) => s.set);
	const qc = useQueryClient();
	const navigate = useNavigate();

	const startMutation = useMutation({
		mutationFn: api.startSession,
		onMutate: () => {
			// Optimistic UX: pretend the snapshot has flipped to "running" while we
			// await the server. Refetched immediately after on success/failure.
			qc.setQueryData(["snapshot"], (prev: unknown) => {
				if (prev && typeof prev === "object") {
					return { ...prev, session_status: "running" };
				}
				return prev;
			});
		},
		onSuccess: (res) => {
			void qc.invalidateQueries({ queryKey: ["snapshot"] });
			if (res.status === "started") {
				toast.success("Session started", {
					description: `Output: ${res.output_dir ?? "—"}`,
				});
				navigate({ to: "/monitor" });
			} else {
				toast.error(
					res.status === "conflict" ? "Already running" : "Bad request",
					{
						description: res.message,
					},
				);
			}
		},
		onError: (e) => {
			void qc.invalidateQueries({ queryKey: ["snapshot"] });
			toast.error("Failed to start session", {
				description: e instanceof Error ? e.message : String(e),
			});
		},
	});

	const wallSecs = estimateWallSeconds(
		config.budget,
		config.seeds,
		config.max_steps,
	);
	const canStart =
		!!config.family &&
		!!config.training_type &&
		!!config.dataset_toml &&
		!!config.output_dir;

	return (
		<div className="flex flex-col gap-8">
			<header className="flex flex-col gap-1">
				<h1 className="text-3xl font-semibold tracking-tight font-display">
					Run
				</h1>
				<p className="text-sm text-muted-foreground">
					Budget the search. Tune finals. Hit start.
				</p>
			</header>

			<Card>
				<CardHeader>
					<CardTitle>Search budget</CardTitle>
					<CardDescription>
						How much compute the orchestrator gets to spend exploring configs.
					</CardDescription>
				</CardHeader>
				<CardContent className="grid gap-5">
					<SliderRow
						label="Candidate configs"
						id="budget"
						min={1}
						max={32}
						value={config.budget}
						onChange={(v) => setField("budget", v)}
						tail={
							<Badge variant="muted" className="gap-1.5">
								<Clock className="h-3 w-3" />
								<span className="font-mono-tight">
									≈ {formatDuration(wallSecs)} rough
								</span>
							</Badge>
						}
					/>
					<SliderRow
						label="Seeds per config"
						id="seeds"
						min={1}
						max={5}
						value={config.seeds}
						onChange={(v) => setField("seeds", v)}
						tail={
							config.seeds >= 2 ? (
								<Badge variant="success">Welch's t-test enabled</Badge>
							) : (
								<Badge variant="muted">no CI at seeds=1</Badge>
							)
						}
					/>
					<SliderRow
						label="Max steps per run"
						id="max_steps"
						min={50}
						max={2000}
						step={50}
						value={config.max_steps}
						onChange={(v) => setField("max_steps", v)}
					/>
					<SliderRow
						label="Wall-time cap (s)"
						id="wall_secs"
						min={60}
						max={7200}
						step={60}
						value={config.wall_secs}
						onChange={(v) => setField("wall_secs", v)}
					/>

					<div className="flex flex-col gap-2">
						<Label>Search method</Label>
						<Tabs
							value={config.search_method}
							onValueChange={(v) => setField("search_method", v)}
							className="w-fit"
						>
							<TabsList>
								<TabsTrigger value="optuna">Optuna TPE</TabsTrigger>
								<TabsTrigger value="random">Random</TabsTrigger>
							</TabsList>
						</Tabs>
					</div>

					<SliderRow
						label="Optuna startup trials"
						id="optuna_startup"
						min={0}
						max={20}
						value={config.optuna_startup}
						onChange={(v) => setField("optuna_startup", v)}
						disabled={config.search_method !== "optuna"}
					/>
					<SliderRow
						label="Curated warm-start (-1 = all)"
						id="n_curated"
						min={-1}
						max={20}
						value={config.n_curated}
						onChange={(v) => setField("n_curated", v)}
					/>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>Finals stage</CardTitle>
					<CardDescription>
						Top-K winners get a longer second-stage training run with extra
						seeds. Top-K = 0 disables.
					</CardDescription>
				</CardHeader>
				<CardContent className="grid gap-5">
					<SliderRow
						label="Top-K (0 = skip)"
						id="finals_top_k"
						min={0}
						max={5}
						value={config.finals_top_k}
						onChange={(v) => setField("finals_top_k", v)}
					/>
					<SliderRow
						label="Finals max steps"
						id="finals_max_steps"
						min={500}
						max={10000}
						step={500}
						value={config.finals_max_steps}
						onChange={(v) => setField("finals_max_steps", v)}
						disabled={config.finals_top_k === 0}
					/>
					<SliderRow
						label="Finals seeds per config"
						id="finals_seeds"
						min={1}
						max={5}
						value={config.finals_seeds}
						onChange={(v) => setField("finals_seeds", v)}
						disabled={config.finals_top_k === 0}
					/>
					<Label className="flex items-start gap-2.5 normal-case tracking-normal cursor-pointer">
						<Checkbox
							id="enable_pairwise_finals"
							checked={config.enable_pairwise_finals}
							disabled={config.finals_top_k === 0}
							onCheckedChange={(v) =>
								setField("enable_pairwise_finals", v === true)
							}
							className="mt-0.5"
						/>
						<div className="flex flex-col gap-0.5">
							<span className="text-sm">Pairwise tournament (Bradley-Terry)</span>
							<span className="text-xs text-muted-foreground">
								After the finals stage, run a round-robin where the VLM
								compares every pair of finalists on every prompt. Adds an
								Elo leaderboard to the report. ~7-10× the finals judge
								load.
							</span>
						</div>
					</Label>
				</CardContent>
			</Card>

			<Card className="border-primary/30 bg-primary/[0.02]">
				<CardContent className="flex flex-wrap items-center gap-4 pt-6">
					<div className="flex flex-col gap-0.5 min-w-0 flex-1">
						<p className="text-xs uppercase tracking-wider text-muted-foreground">
							Estimated wall-time
						</p>
						<p className="font-mono-tight text-2xl">
							{formatDuration(wallSecs)}
						</p>
						<p className="text-xs text-muted-foreground">
							budget × seeds × max-steps × ~2.85s/it
						</p>
					</div>
					<Button
						size="lg"
						disabled={!canStart || startMutation.isPending}
						onClick={() => startMutation.mutate(config)}
						className="gap-2"
					>
						<Play className="h-4 w-4 fill-current" />
						{startMutation.isPending
							? "Queueing setup…"
							: "Start orchestration"}
					</Button>
				</CardContent>
			</Card>

			{!canStart && (
				<p className="text-xs text-muted-foreground text-center">
					Pick a model preset, dataset TOML, and output directory in{" "}
					<strong>Setup</strong> before starting.
				</p>
			)}
		</div>
	);
}

function SliderRow({
	label,
	id,
	min,
	max,
	step = 1,
	value,
	onChange,
	disabled,
	tail,
}: {
	label: string;
	id: string;
	min: number;
	max: number;
	step?: number;
	value: number;
	onChange: (v: number) => void;
	disabled?: boolean;
	tail?: React.ReactNode;
}) {
	return (
		<div className="flex flex-col gap-2">
			<div className="flex items-center justify-between gap-3">
				<Label htmlFor={id}>{label}</Label>
				{tail}
			</div>
			<div className="flex items-center gap-3">
				<Slider
					id={id}
					min={min}
					max={max}
					step={step}
					value={[value]}
					onValueChange={(v) => onChange(v[0] ?? min)}
					disabled={disabled}
					className="flex-1"
				/>
				<Input
					type="number"
					min={min}
					max={max}
					step={step}
					value={value}
					onChange={(e) => onChange(Number(e.target.value))}
					disabled={disabled}
					className="w-24 text-right"
				/>
			</div>
		</div>
	);
}
