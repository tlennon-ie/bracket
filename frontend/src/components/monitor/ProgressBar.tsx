import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface Props {
	pct: number;
	completed: number;
	total: number;
	className?: string;
}

export function ProgressBar({ pct, completed, total, className }: Props) {
	const safe = Number.isFinite(pct) ? Math.max(0, Math.min(100, pct)) : 0;
	return (
		<div className={cn("flex flex-col gap-1.5", className)}>
			<div className="flex items-baseline justify-between text-xs text-muted-foreground">
				<span className="uppercase tracking-wider">Progress</span>
				<span className="font-mono-tight">
					{completed}/{total} runs · {safe.toFixed(1)}%
				</span>
			</div>
			<Progress
				value={safe}
				indicatorClassName={safe >= 100 ? "bg-success" : "bg-primary"}
			/>
		</div>
	);
}
