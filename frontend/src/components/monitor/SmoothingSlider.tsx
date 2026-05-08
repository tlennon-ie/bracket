import { Slider } from "@/components/ui/slider";
import { useUIStore } from "@/store/ui";

export function SmoothingSlider() {
	const smoothing = useUIStore((s) => s.smoothing);
	const setSmoothing = useUIStore((s) => s.setSmoothing);
	return (
		<div className="flex items-center gap-3">
			<label
				htmlFor="smoothing"
				className="text-[11px] uppercase tracking-wider text-muted-foreground shrink-0"
			>
				Smoothing
			</label>
			<Slider
				id="smoothing"
				min={0}
				max={0.99}
				step={0.01}
				value={[smoothing]}
				onValueChange={(v) => setSmoothing(v[0] ?? 0)}
				aria-label="EMA smoothing alpha"
				className="w-40"
			/>
			<span className="font-mono-tight text-xs text-foreground tabular-nums w-10">
				{smoothing.toFixed(2)}
			</span>
		</div>
	);
}
