// Number / duration / score formatting helpers — keep dumb & pure.

export function formatScore(
	value: number | null | undefined,
	digits = 4,
): string {
	if (value === null || value === undefined || Number.isNaN(value)) return "—";
	return value.toFixed(digits);
}

export function formatPct(
	value: number | null | undefined,
	digits = 1,
): string {
	if (value === null || value === undefined || Number.isNaN(value)) return "—";
	return `${value.toFixed(digits)}%`;
}

export function formatDuration(seconds: number | null | undefined): string {
	if (seconds === null || seconds === undefined || Number.isNaN(seconds))
		return "—";
	const s = Math.max(0, Math.floor(seconds));
	if (s < 60) return `${s}s`;
	const m = Math.floor(s / 60);
	const rs = s % 60;
	if (m < 60) return `${m}m ${rs.toString().padStart(2, "0")}s`;
	const h = Math.floor(m / 60);
	const rm = m % 60;
	return `${h}h ${rm.toString().padStart(2, "0")}m`;
}

export function formatStep(step: number | null | undefined): string {
	if (step === null || step === undefined) return "—";
	return step.toLocaleString();
}

export function shortenRunId(runId: string, maxLen = 18): string {
	if (runId.length <= maxLen) return runId;
	return `${runId.slice(0, 6)}…${runId.slice(-8)}`;
}

export function estimateWallSeconds(
	budget: number,
	seeds: number,
	maxSteps: number,
	secondsPerIt = 2.85,
): number {
	return budget * seeds * maxSteps * secondsPerIt;
}
