// PromoteDialog — pre-flight for "Train full" on a search-stage candidate.
// Collects max_steps, save cadence, save_state, sample cadence, and an
// optional resume_from path; submits to /api/runs/{run_id}/promote. The
// active orchestration session is replaced by the promoted run, so the
// session must be done (or stopped) before this can succeed.

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import type { CandidateRow } from "@/types/domain";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";

interface PromoteDialogProps {
	row: CandidateRow | null;
	open: boolean;
	onOpenChange: (open: boolean) => void;
}

interface FormState {
	max_steps: number;
	save_every_n_steps: number;
	sample_every_n_steps: number | null;
	save_state: boolean;
	resume_from: string;
	full_dataset_toml: string;
}

const DEFAULTS: FormState = {
	max_steps: 2000,
	save_every_n_steps: 500,
	sample_every_n_steps: null,
	save_state: true,
	resume_from: "",
	full_dataset_toml: "",
};

export function PromoteDialog({ row, open, onOpenChange }: PromoteDialogProps) {
	const [form, setForm] = useState<FormState>(DEFAULTS);
	const qc = useQueryClient();
	const navigate = useNavigate();

	const mutation = useMutation({
		mutationFn: () => {
			if (!row) throw new Error("no row");
			return api.promoteRun(row.run_id, {
				max_steps: form.max_steps,
				save_every_n_steps: form.save_every_n_steps,
				sample_every_n_steps: form.sample_every_n_steps,
				save_state: form.save_state,
				resume_from: form.resume_from.trim(),
				full_dataset_toml: form.full_dataset_toml.trim(),
			});
		},
		onSuccess: (res) => {
			if (res.status === "started") {
				toast.success("Promoted run started", {
					description: res.message,
				});
				onOpenChange(false);
				qc.invalidateQueries({ queryKey: ["snapshot"] });
				qc.invalidateQueries({ queryKey: ["runs"] });
				void navigate({ to: "/monitor" });
			} else {
				toast.error("Promote failed", { description: res.message });
			}
		},
		onError: (e) => {
			toast.error("Promote failed", {
				description: e instanceof Error ? e.message : String(e),
			});
		},
	});

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="max-w-lg">
				<DialogHeader>
					<DialogTitle>Promote to full training run</DialogTitle>
					<DialogDescription>
						Re-runs <code className="font-mono-tight">{row?.run_id ?? "—"}</code>{" "}
						with its winning hyperparameters against the full dataset (not the
						search subset). Replaces the active session — make sure the
						current one is stopped or done.
					</DialogDescription>
				</DialogHeader>

				<div className="grid grid-cols-2 gap-4 py-2">
					<div className="flex flex-col gap-1.5">
						<Label htmlFor="promote_max_steps">Max steps</Label>
						<Input
							id="promote_max_steps"
							type="number"
							min={1}
							value={form.max_steps}
							onChange={(e) =>
								setForm((s) => ({
									...s,
									max_steps: Number.parseInt(e.target.value, 10) || 0,
								}))
							}
						/>
					</div>
					<div className="flex flex-col gap-1.5">
						<Label htmlFor="promote_save_every">Save every N steps</Label>
						<Input
							id="promote_save_every"
							type="number"
							min={1}
							value={form.save_every_n_steps}
							onChange={(e) =>
								setForm((s) => ({
									...s,
									save_every_n_steps:
										Number.parseInt(e.target.value, 10) || 0,
								}))
							}
						/>
					</div>
					<div className="flex flex-col gap-1.5 col-span-2">
						<Label htmlFor="promote_sample_every">
							Sample every N steps (optional)
						</Label>
						<Input
							id="promote_sample_every"
							type="number"
							min={0}
							placeholder="leave blank to match save cadence"
							value={form.sample_every_n_steps ?? ""}
							onChange={(e) => {
								const v = e.target.value;
								setForm((s) => ({
									...s,
									sample_every_n_steps:
										v === "" ? null : Number.parseInt(v, 10) || null,
								}));
							}}
						/>
					</div>
					<div className="flex flex-col gap-1.5 col-span-2">
						<Label htmlFor="promote_resume">Resume from (optional)</Label>
						<Input
							id="promote_resume"
							placeholder="path to prior LoRA or training state dir"
							value={form.resume_from}
							onChange={(e) =>
								setForm((s) => ({ ...s, resume_from: e.target.value }))
							}
						/>
					</div>
					<div className="flex flex-col gap-1.5 col-span-2">
						<Label htmlFor="promote_dataset">
							Full dataset TOML override (optional)
						</Label>
						<Input
							id="promote_dataset"
							placeholder="leave blank to use the session's original dataset"
							value={form.full_dataset_toml}
							onChange={(e) =>
								setForm((s) => ({ ...s, full_dataset_toml: e.target.value }))
							}
						/>
					</div>
					<Label
						className="col-span-2 flex items-start gap-2 normal-case tracking-normal cursor-pointer"
					>
						<Checkbox
							checked={form.save_state}
							onCheckedChange={(v) =>
								setForm((s) => ({ ...s, save_state: v === true }))
							}
							className="mt-0.5"
						/>
						<div className="flex flex-col gap-0.5">
							<span className="text-sm">Save training state at each checkpoint</span>
							<span className="text-xs text-muted-foreground">
								Emits <code className="font-mono-tight text-[11px]">--save_state</code>{" "}
								so optimizer + RNG + scheduler state are saved alongside
								weights. Required if you want to resume later.
							</span>
						</div>
					</Label>
				</div>

				<DialogFooter>
					<Button
						variant="ghost"
						onClick={() => onOpenChange(false)}
						disabled={mutation.isPending}
					>
						Cancel
					</Button>
					<Button
						onClick={() => mutation.mutate()}
						disabled={!row || mutation.isPending}
					>
						{mutation.isPending ? "Starting…" : "Start full training"}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}
