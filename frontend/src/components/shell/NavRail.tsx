import { BrandMark } from "@/components/shell/BrandMark";
import { ThemeToggle } from "@/components/shell/ThemeToggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
	Tooltip,
	TooltipContent,
	TooltipProvider,
	TooltipTrigger,
} from "@/components/ui/tooltip";
import { useSnapshot } from "@/hooks/useSnapshot";
import { cn } from "@/lib/utils";
import { useUIStore } from "@/store/ui";
import { Link, useRouterState } from "@tanstack/react-router";
import {
	Activity,
	ChevronLeft,
	ChevronRight,
	FileText,
	Play,
	Settings2,
} from "lucide-react";
import type { ComponentType, SVGProps } from "react";

type IconType = ComponentType<SVGProps<SVGSVGElement>>;

interface NavItem {
	to: string;
	label: string;
	icon: IconType;
	description: string;
}

const NAV: NavItem[] = [
	{
		to: "/setup",
		label: "Setup",
		icon: Settings2,
		description: "Pick model, dataset, judge",
	},
	{
		to: "/run",
		label: "Run",
		icon: Play,
		description: "Budget, finals, start",
	},
	{
		to: "/monitor",
		label: "Monitor",
		icon: Activity,
		description: "Live loss, gallery, status",
	},
	{
		to: "/results",
		label: "Results",
		icon: FileText,
		description: "Report, ledger, comparison",
	},
];

export function NavRail() {
	const collapsed = useUIStore((s) => s.sidebarCollapsed);
	const toggle = useUIStore((s) => s.toggleSidebar);
	const { snapshot, wsStatus } = useSnapshot();
	const status = snapshot?.session_status ?? "idle";
	const pathname = useRouterState({ select: (s) => s.location.pathname });

	const isLive = status === "running" || status === "stopping";

	return (
		<TooltipProvider delayDuration={150}>
			<nav
				aria-label="Primary navigation"
				className={cn(
					"flex flex-col h-full border-r border-border bg-card transition-[width] duration-150 ease-out",
					collapsed ? "w-16" : "w-60",
				)}
			>
				<div
					className={cn(
						"flex items-center px-4 py-4 border-b border-border",
						collapsed && "justify-center px-0",
					)}
				>
					<BrandMark collapsed={collapsed} />
				</div>

				<ul className="flex-1 flex flex-col gap-1 px-2 py-4">
					{NAV.map((item) => {
						const active =
							pathname === item.to ||
							(item.to === "/monitor" && pathname.startsWith("/monitor")) ||
							(item.to === "/results" && pathname.startsWith("/results"));
						const showLive = item.to === "/monitor" && isLive;
						return (
							<li key={item.to}>
								<Tooltip>
									<TooltipTrigger asChild>
										<Link
											to={item.to}
											className={cn(
												"group flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
												"hover:bg-accent hover:text-accent-foreground",
												active &&
													"bg-accent text-accent-foreground font-medium",
												collapsed && "justify-center px-0",
											)}
										>
											<item.icon className="h-4 w-4 shrink-0" />
											{!collapsed && (
												<span className="flex-1">{item.label}</span>
											)}
											{showLive && (
												<span
													className={cn(
														"pulse-dot rounded-full bg-success",
														collapsed
															? "h-1.5 w-1.5 absolute right-2 top-2"
															: "h-2 w-2",
													)}
												/>
											)}
										</Link>
									</TooltipTrigger>
									{collapsed && (
										<TooltipContent side="right">
											<div className="flex flex-col gap-0.5">
												<span className="font-medium">{item.label}</span>
												<span className="text-muted-foreground">
													{item.description}
												</span>
											</div>
										</TooltipContent>
									)}
								</Tooltip>
							</li>
						);
					})}
				</ul>

				<div
					className={cn("flex flex-col gap-2 px-3 py-3 border-t border-border")}
				>
					{!collapsed && (
						<div className="flex items-center justify-between text-[11px]">
							<span className="text-muted-foreground uppercase tracking-wider">
								Status
							</span>
							<Badge
								variant={
									status === "running"
										? "success"
										: status === "error"
											? "destructive"
											: status === "done"
												? "default"
												: "muted"
								}
								className="uppercase text-[10px]"
							>
								{status}
							</Badge>
						</div>
					)}
					{!collapsed && (
						<div className="flex items-center justify-between text-[10px] text-muted-foreground">
							<span className="uppercase tracking-wider">WS</span>
							<span
								className={cn(
									"font-mono-tight",
									wsStatus === "open" && "text-success",
									wsStatus === "connecting" && "text-warning",
									(wsStatus === "closed" || wsStatus === "error") &&
										"text-destructive",
								)}
							>
								{wsStatus}
							</span>
						</div>
					)}
					<div
						className={cn("flex items-center gap-2", collapsed && "flex-col")}
					>
						<ThemeToggle collapsed={collapsed} />
						<Button
							variant="ghost"
							size="icon"
							onClick={toggle}
							aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
							className="ml-auto"
						>
							{collapsed ? (
								<ChevronRight className="h-4 w-4" />
							) : (
								<ChevronLeft className="h-4 w-4" />
							)}
						</Button>
					</div>
				</div>
			</nav>
		</TooltipProvider>
	);
}
