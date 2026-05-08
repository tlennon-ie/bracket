// Theme bootstrap — applies `.dark` class to <html> based on the persisted
// preference and `prefers-color-scheme`. Also responds to system changes.

import { useUIStore } from "@/store/ui";
import { useEffect } from "react";

function resolveTheme(stored: "light" | "dark" | "system"): "light" | "dark" {
	if (stored === "system") {
		return typeof window !== "undefined" &&
			window.matchMedia("(prefers-color-scheme: dark)").matches
			? "dark"
			: "light";
	}
	return stored;
}

export function useTheme() {
	const theme = useUIStore((s) => s.theme);
	const setTheme = useUIStore((s) => s.setTheme);

	useEffect(() => {
		const apply = () => {
			const resolved = resolveTheme(theme);
			const root = document.documentElement;
			root.classList.toggle("dark", resolved === "dark");
			root.dataset.theme = resolved;
		};
		apply();

		if (theme !== "system" || typeof window === "undefined") return;
		const mq = window.matchMedia("(prefers-color-scheme: dark)");
		const listener = () => apply();
		mq.addEventListener("change", listener);
		return () => mq.removeEventListener("change", listener);
	}, [theme]);

	return { theme, setTheme, resolved: resolveTheme(theme) };
}
