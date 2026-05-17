// Session-config draft state. Survives navigation between /setup and /run.
// Persisted to localStorage so a refresh doesn't wipe the user's typing.

import type { StartSessionRequest } from "@/types/domain";
import { create } from "zustand";
import { persist } from "zustand/middleware";

const DEFAULTS: StartSessionRequest = {
	family: "",
	training_type: "",
	dataset_toml: "",
	output_dir: "",
	sample_prompts: "",
	sample_prompts_ood: null,
	resume: "",
	images_per_dataset: 12,
	vram_gb: null,
	budget: 8,
	max_steps: 300,
	wall_secs: 1800,
	seeds: 1,
	search_method: "optuna",
	optuna_startup: 5,
	n_curated: -1,
	finals_top_k: 0,
	finals_max_steps: 2000,
	finals_seeds: 2,
	judge_method: "none",
	judge_base_url: "http://localhost:1234/v1",
	judge_model: "qwen3-vl-8b-thinking-abliterated",
	judge_loss_weight: 0.3,
	judge_sample_weight: 0.7,
	judge_disable_thinking: false,
	judge_n_samples: 1,
	enable_clip_iqa_gate: false,
	clip_iqa_dq_threshold: 0.3,
	lr_min: null,
	lr_max: null,
	batch_size_min: null,
	batch_size_max: null,
	gradient_checkpointing_mode: null,
	use_history_priors: false,
	preset_field_values: {},
};

interface SessionConfigStore {
	config: StartSessionRequest;
	set: <K extends keyof StartSessionRequest>(
		key: K,
		value: StartSessionRequest[K],
	) => void;
	setPresetField: (name: string, value: string) => void;
	setFamilyType: (family: string, type: string) => void;
	reset: () => void;
}

export const useSessionConfigStore = create<SessionConfigStore>()(
	persist(
		(set) => ({
			config: { ...DEFAULTS },
			set: (key, value) =>
				set((state) => ({ config: { ...state.config, [key]: value } })),
			setPresetField: (name, value) =>
				set((state) => ({
					config: {
						...state.config,
						preset_field_values: {
							...state.config.preset_field_values,
							[name]: value,
						},
					},
				})),
			setFamilyType: (family, training_type) =>
				set((state) => ({
					config: {
						...state.config,
						family,
						training_type,
						// Reset preset-specific values when switching presets.
						preset_field_values:
							state.config.family === family &&
							state.config.training_type === training_type
								? state.config.preset_field_values
								: {},
					},
				})),
			reset: () => set({ config: { ...DEFAULTS } }),
		}),
		{ name: "bracket-session-config" },
	),
);
