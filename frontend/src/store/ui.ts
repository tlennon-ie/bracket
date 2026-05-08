// Ephemeral UI state — anything not server-derived.
// Persisted slices (smoothing, theme, sidebar) hit localStorage; comparison
// selection is reflected in URL search params via the Results route.

import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Theme = "light" | "dark" | "system";

interface UIStore {
	smoothing: number; // EMA alpha, 0..0.99
	setSmoothing: (v: number) => void;

	sidebarCollapsed: boolean;
	toggleSidebar: () => void;
	setSidebarCollapsed: (v: boolean) => void;

	theme: Theme;
	setTheme: (t: Theme) => void;

	judgeWeights: { loss: number; sample: number };
	setJudgeWeights: (w: { loss: number; sample: number }) => void;

	/** Clamps to ≤3, keeps the array stable. */
	compareRunIds: string[];
	toggleCompareRun: (id: string) => void;
	setCompareRuns: (ids: string[]) => void;
	clearCompareRuns: () => void;
}

export const useUIStore = create<UIStore>()(
	persist(
		(set, get) => ({
			smoothing: 0.6,
			setSmoothing: (v) => set({ smoothing: Math.max(0, Math.min(0.99, v)) }),

			sidebarCollapsed: false,
			toggleSidebar: () => set({ sidebarCollapsed: !get().sidebarCollapsed }),
			setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),

			theme: "system",
			setTheme: (t) => set({ theme: t }),

			judgeWeights: { loss: 0.3, sample: 0.7 },
			setJudgeWeights: (w) => {
				const total = w.loss + w.sample;
				if (total <= 0) return;
				set({
					judgeWeights: { loss: w.loss / total, sample: w.sample / total },
				});
			},

			compareRunIds: [],
			toggleCompareRun: (id) => {
				const cur = get().compareRunIds;
				if (cur.includes(id)) {
					set({ compareRunIds: cur.filter((x) => x !== id) });
				} else if (cur.length < 3) {
					set({ compareRunIds: [...cur, id] });
				}
			},
			setCompareRuns: (ids) => set({ compareRunIds: ids.slice(0, 3) }),
			clearCompareRuns: () => set({ compareRunIds: [] }),
		}),
		{
			name: "bracket-ui",
			partialize: (state) => ({
				smoothing: state.smoothing,
				sidebarCollapsed: state.sidebarCollapsed,
				theme: state.theme,
				judgeWeights: state.judgeWeights,
			}),
		},
	),
);
