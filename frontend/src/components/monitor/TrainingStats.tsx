// TrainingStats — top-level statistic strip rendered above the loss chart.
// Three columns: current step, throughput (it/s or s/it), and ETA. All three
// gracefully render an em-dash when the underlying field is null (e.g. before
// the first tfevents sample lands).

import {
	formatDuration,
	formatEta,
	formatStep,
	formatStepsPerSec,
} from "@/lib/format";

interface TrainingStatsProps {
	stepsDone: number | null;
	maxSteps: number | null;
	stepsPerSec: number | null;
	elapsedS: number;
}

export function TrainingStats({
	stepsDone,
	maxSteps,
	stepsPerSec,
	elapsedS,
}: TrainingStatsProps) {
	const stepsRemaining =
		stepsDone !== null && maxSteps !== null
			? Math.max(0, maxSteps - stepsDone)
			: null;
	const stepText =
		stepsDone !== null && maxSteps !== null
			? `${formatStep(stepsDone)} / ${formatStep(maxSteps)}`
			: formatStep(stepsDone);

	return (
		<div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
			<Stat label="step" value={stepText} />
			<Stat label="throughput" value={formatStepsPerSec(stepsPerSec)} />
			<Stat label="eta" value={formatEta(stepsRemaining, stepsPerSec)} />
			<Stat label="elapsed" value={formatDuration(elapsedS)} />
		</div>
	);
}

function Stat({ label, value }: { label: string; value: string }) {
	return (
		<div className="flex flex-col gap-0.5 rounded-md border border-border bg-card/40 px-3 py-2">
			<span className="text-[10px] uppercase tracking-wider text-muted-foreground">
				{label}
			</span>
			<span className="text-base font-medium font-mono-tight tabular-nums">
				{value}
			</span>
		</div>
	);
}
