import { api } from "@/lib/api";
import { createFileRoute, redirect } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
	beforeLoad: async () => {
		try {
			const snap = await api.getSnapshot(true);
			if (
				snap &&
				(snap.session_status === "running" ||
					snap.session_status === "stopping")
			) {
				throw redirect({ to: "/monitor" });
			}
		} catch (e) {
			if (e && typeof e === "object" && "to" in e) throw e; // re-throw redirect
			// network error → fall through to /setup
		}
		throw redirect({ to: "/setup" });
	},
});
