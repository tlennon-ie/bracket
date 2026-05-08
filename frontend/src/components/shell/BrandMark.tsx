import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";

interface BrandMarkProps {
	collapsed?: boolean;
	className?: string;
}

export function BrandMark({ collapsed, className }: BrandMarkProps) {
	const { resolved } = useTheme();
	const src = resolved === "dark" ? "/logo-dark.png" : "/logo.png";
	return (
		<div className={cn("flex items-center gap-3 select-none", className)}>
			<img
				src={src}
				alt="Bracket"
				width={28}
				height={28}
				className="h-7 w-7 object-contain"
				draggable={false}
			/>
			{!collapsed && (
				<div className="flex flex-col leading-none">
					<span className="text-base font-semibold tracking-tight font-display">
						bracket
					</span>
					<span className="text-[10px] text-muted-foreground tracking-widest uppercase mt-0.5 font-mono-tight">
						v0.1
					</span>
				</div>
			)}
		</div>
	);
}
