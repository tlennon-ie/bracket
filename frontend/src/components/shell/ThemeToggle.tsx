import { Button } from "@/components/ui/button";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { Monitor, Moon, Sun } from "lucide-react";

const ORDER = ["light", "dark", "system"] as const;

export function ThemeToggle({ collapsed }: { collapsed?: boolean }) {
	const { theme, setTheme } = useTheme();
	const next = () => {
		const idx = ORDER.indexOf(theme as (typeof ORDER)[number]);
		setTheme(ORDER[(idx + 1) % ORDER.length] ?? "system");
	};
	const Icon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;
	return (
		<Button
			variant="ghost"
			size={collapsed ? "icon" : "sm"}
			onClick={next}
			aria-label={`Theme: ${theme}. Click to cycle.`}
			className={cn("justify-start gap-2", collapsed && "justify-center")}
		>
			<Icon className="h-4 w-4" />
			{!collapsed && <span className="capitalize text-xs">{theme}</span>}
		</Button>
	);
}
