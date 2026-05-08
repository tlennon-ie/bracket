import { AppShell } from "@/components/shell/AppShell";
import { Outlet, createRootRoute } from "@tanstack/react-router";

export const Route = createRootRoute({
	component: () => (
		<AppShell>
			<Outlet />
		</AppShell>
	),
});
