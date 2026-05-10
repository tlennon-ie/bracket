import { DatasetTomlDrop } from "@/components/setup/DatasetTomlDrop";
import { DynamicFieldList } from "@/components/setup/DynamicFieldList";
import { ModelFamilyPicker } from "@/components/setup/ModelFamilyPicker";
import { Badge } from "@/components/ui/badge";
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
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { useFamilies, usePreset } from "@/hooks/usePresets";
import { useSessionConfigStore } from "@/store/sessionConfig";
import type { FieldSpec } from "@/types/domain";
import { createFileRoute } from "@tanstack/react-router";
import { AlertCircle, BookOpen } from "lucide-react";
import { useMemo } from "react";

export const Route = createFileRoute("/setup")({
	component: SetupPage,
});

function SetupPage() {
	const families = useFamilies();
	const config = useSessionConfigStore((s) => s.config);
	const setField = useSessionConfigStore((s) => s.set);
	const setPresetField = useSessionConfigStore((s) => s.setPresetField);
	const setFamilyType = useSessionConfigStore((s) => s.setFamilyType);

	const preset = usePreset(config.family || null, config.training_type || null);

	const sessionFields = useMemo<FieldSpec[]>(
		() => preset.data?.session_fields ?? [],
		[preset.data],
	);

	// ── Empty-state: API unreachable ──────────────────────────────────────────
	if (families.isError) {
		return (
			<Card className="max-w-xl mx-auto mt-12">
				<CardContent className="flex items-start gap-3 p-6">
					<AlertCircle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
					<div className="flex flex-col gap-1">
						<p className="font-medium text-sm">Bracket API unreachable</p>
						<p className="text-xs text-muted-foreground">
							No response from{" "}
							<code className="font-mono-tight">/api/presets/families</code>.
							Start the backend with{" "}
							<code className="font-mono-tight">bracket-serve --port 8000</code>{" "}
							and refresh.
						</p>
					</div>
				</CardContent>
			</Card>
		);
	}

	return (
		<div className="flex flex-col gap-8">
			<header className="flex flex-col gap-1">
				<h1 className="text-3xl font-semibold tracking-tight font-display">
					Setup
				</h1>
				<p className="text-sm text-muted-foreground italic">
					Train the same diffusion model eight ways. Pick the one that looks
					best. With a p-value.
				</p>
			</header>

			<Card>
				<CardHeader>
					<CardTitle>Preset</CardTitle>
					<CardDescription>
						Pick a model family then a training type. Fields update from the
						registry.
					</CardDescription>
				</CardHeader>
				<CardContent>
					<ModelFamilyPicker
						family={config.family || null}
						trainingType={config.training_type || null}
						onChange={(fam, t) => setFamilyType(fam ?? "", t ?? "")}
					/>
				</CardContent>
			</Card>

			{preset.isLoading && config.family && config.training_type && (
				<div className="grid gap-4">
					<Skeleton className="h-32 w-full" />
					<Skeleton className="h-48 w-full" />
				</div>
			)}

			{preset.data && (
				<>
					<Card>
						<CardHeader className="flex-row items-start gap-3 space-y-0">
							<BookOpen className="h-5 w-5 mt-0.5 text-muted-foreground" />
							<div className="flex-1">
								<CardTitle>{preset.data.display_name}</CardTitle>
								<CardDescription>
									Auto pre-cache:{" "}
									<Badge
										variant={preset.data.needs_pre_cache ? "default" : "muted"}
										className="ml-1"
									>
										{preset.data.needs_pre_cache ? "yes" : "no"}
									</Badge>
								</CardDescription>
							</div>
						</CardHeader>
						{preset.data.notes && (
							<CardContent>
								<pre className="whitespace-pre-wrap text-xs text-muted-foreground font-mono-tight">
									{preset.data.notes}
								</pre>
							</CardContent>
						)}
					</Card>

					<Card>
						<CardHeader>
							<CardTitle>Model paths</CardTitle>
							<CardDescription>
								Required by the trainer adapter.
							</CardDescription>
						</CardHeader>
						<CardContent>
							<DynamicFieldList
								fields={preset.data.fields}
								values={config.preset_field_values}
								onChange={setPresetField}
							/>
						</CardContent>
					</Card>
				</>
			)}

			<Card>
				<CardHeader>
					<CardTitle>Dataset & session</CardTitle>
					<CardDescription>
						Drop a TOML to preview. Type the path the backend should read from
						below.
					</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-4">
					<DatasetTomlDrop />

					<div className="grid gap-4">
						<FieldRow label="Dataset config TOML *" htmlFor="dataset_toml">
							<Input
								id="dataset_toml"
								value={config.dataset_toml}
								placeholder="./configs/dataset.toml"
								onChange={(e) => setField("dataset_toml", e.target.value)}
							/>
						</FieldRow>
						<FieldRow label="Session output directory *" htmlFor="output_dir">
							<Input
								id="output_dir"
								value={config.output_dir}
								placeholder="./runs/session-001"
								onChange={(e) => setField("output_dir", e.target.value)}
							/>
						</FieldRow>
						<FieldRow
							label="Sample prompts (optional)"
							htmlFor="sample_prompts"
						>
							<Input
								id="sample_prompts"
								value={config.sample_prompts}
								placeholder="./configs/sample_prompts.txt"
								onChange={(e) => setField("sample_prompts", e.target.value)}
							/>
						</FieldRow>
						<FieldRow label="Resume from (optional)" htmlFor="resume">
							<Input
								id="resume"
								value={config.resume}
								placeholder="path to a previous run.toml"
								onChange={(e) => setField("resume", e.target.value)}
							/>
						</FieldRow>
					</div>

					{sessionFields.length > 0 && (
						<DynamicFieldList
							fields={sessionFields.filter(
								(f) =>
									![
										"dataset_toml",
										"output_dir",
										"sample_prompts",
										"resume",
									].includes(f.name),
							)}
							values={config.preset_field_values}
							onChange={setPresetField}
						/>
					)}

					<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
						<FieldRow label="Subset images per dataset" htmlFor="images">
							<div className="flex items-center gap-3">
								<Slider
									id="images"
									min={4}
									max={64}
									step={2}
									value={[config.images_per_dataset]}
									onValueChange={(v) =>
										setField("images_per_dataset", v[0] ?? 12)
									}
									className="flex-1"
								/>
								<span className="font-mono-tight text-sm tabular-nums w-8 text-right">
									{config.images_per_dataset}
								</span>
							</div>
						</FieldRow>
						<FieldRow label="VRAM override (GB)" htmlFor="vram">
							<Input
								id="vram"
								type="number"
								min={0}
								step={1}
								value={config.vram_gb ?? ""}
								placeholder="auto-detect"
								onChange={(e) =>
									setField(
										"vram_gb",
										e.target.value === "" ? null : Number(e.target.value),
									)
								}
							/>
						</FieldRow>
					</div>
				</CardContent>
			</Card>

			<Card>
				<CardHeader>
					<CardTitle>VLM judge (optional)</CardTitle>
					<CardDescription>
						Augment loss with a vision model's read on sample quality.
					</CardDescription>
				</CardHeader>
				<CardContent className="flex flex-col gap-4">
					<RadioGroup
						value={config.judge_method}
						onValueChange={(v) => setField("judge_method", v)}
						className="grid grid-cols-2 gap-2 max-w-sm"
					>
						<Label className="flex items-center gap-2 normal-case tracking-normal">
							<RadioGroupItem value="none" />
							<span className="text-sm">None</span>
						</Label>
						<Label className="flex items-center gap-2 normal-case tracking-normal">
							<RadioGroupItem value="lmstudio" />
							<span className="text-sm">LM Studio</span>
						</Label>
					</RadioGroup>
					<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
						<FieldRow label="Judge base URL" htmlFor="judge_url">
							<Input
								id="judge_url"
								value={config.judge_base_url}
								disabled={config.judge_method === "none"}
								onChange={(e) => setField("judge_base_url", e.target.value)}
							/>
						</FieldRow>
						<FieldRow label="Judge model" htmlFor="judge_model">
							<Input
								id="judge_model"
								value={config.judge_model}
								disabled={config.judge_method === "none"}
								onChange={(e) => setField("judge_model", e.target.value)}
							/>
						</FieldRow>
					</div>
					<div className="grid grid-cols-2 gap-4">
						<FieldRow label="Loss weight" htmlFor="loss_w">
							<div className="flex items-center gap-3">
								<Slider
									id="loss_w"
									min={0}
									max={1}
									step={0.05}
									value={[config.judge_loss_weight]}
									onValueChange={(v) => {
										const lw = v[0] ?? 0.3;
										setField("judge_loss_weight", lw);
										setField(
											"judge_sample_weight",
											Number((1 - lw).toFixed(2)),
										);
									}}
									className="flex-1"
								/>
								<span className="font-mono-tight text-sm tabular-nums w-10 text-right">
									{config.judge_loss_weight.toFixed(2)}
								</span>
							</div>
						</FieldRow>
						<FieldRow label="Sample weight" htmlFor="sample_w">
							<div className="flex items-center gap-3">
								<Slider
									id="sample_w"
									min={0}
									max={1}
									step={0.05}
									value={[config.judge_sample_weight]}
									onValueChange={(v) => {
										const sw = v[0] ?? 0.7;
										setField("judge_sample_weight", sw);
										setField("judge_loss_weight", Number((1 - sw).toFixed(2)));
									}}
									className="flex-1"
								/>
								<span className="font-mono-tight text-sm tabular-nums w-10 text-right">
									{config.judge_sample_weight.toFixed(2)}
								</span>
							</div>
						</FieldRow>
					</div>
					<Label className="flex items-start gap-2.5 normal-case tracking-normal cursor-pointer">
						<Checkbox
							id="judge_disable_thinking"
							checked={config.judge_disable_thinking}
							disabled={config.judge_method === "none"}
							onCheckedChange={(v) =>
								setField("judge_disable_thinking", v === true)
							}
							className="mt-0.5"
						/>
						<div className="flex flex-col gap-0.5">
							<span className="text-sm">Disable thinking mode</span>
							<span className="text-xs text-muted-foreground">
								Sends{" "}
								<code className="font-mono-tight text-[11px]">
									chat_template_kwargs={"{"}enable_thinking: false{"}"}
								</code>{" "}
								to LM Studio. Skips the{" "}
								<code className="font-mono-tight text-[11px]">
									&lt;think&gt;
								</code>{" "}
								preamble on Qwen3-class VLMs — roughly halves judge latency
								and eliminates the prose-overflow failure mode. Templates
								without that variable ignore it, so it's safe to leave on.
							</span>
						</div>
					</Label>
				</CardContent>
			</Card>
		</div>
	);
}

function FieldRow({
	label,
	htmlFor,
	children,
}: {
	label: string;
	htmlFor: string;
	children: React.ReactNode;
}) {
	return (
		<div className="flex flex-col gap-1.5">
			<Label htmlFor={htmlFor}>{label}</Label>
			{children}
		</div>
	);
}
