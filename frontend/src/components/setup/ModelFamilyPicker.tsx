import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { useFamilies, useTrainingTypes } from "@/hooks/usePresets";
import { Loader2 } from "lucide-react";

interface Props {
	family: string | null;
	trainingType: string | null;
	onChange: (family: string | null, type: string | null) => void;
}

export function ModelFamilyPicker({ family, trainingType, onChange }: Props) {
	const families = useFamilies();
	const types = useTrainingTypes(family);

	return (
		<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div className="flex flex-col gap-1.5">
				<Label htmlFor="family-select">Model family</Label>
				<Select
					value={family ?? undefined}
					onValueChange={(v) => onChange(v, null)}
				>
					<SelectTrigger id="family-select" aria-label="Model family">
						<SelectValue placeholder="Pick a model family…" />
					</SelectTrigger>
					<SelectContent>
						{families.isLoading && (
							<div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
								<Loader2 className="h-3 w-3 animate-spin" />
								Loading families…
							</div>
						)}
						{families.data?.map((f) => (
							<SelectItem key={f.name} value={f.name}>
								{f.label}
							</SelectItem>
						))}
						{families.isError && (
							<div className="px-3 py-2 text-xs text-destructive">
								Failed to load families. Is the API up?
							</div>
						)}
					</SelectContent>
				</Select>
			</div>

			<div className="flex flex-col gap-1.5">
				<Label htmlFor="type-select">Training type</Label>
				<Select
					value={trainingType ?? undefined}
					onValueChange={(v) => onChange(family, v)}
					disabled={!family}
				>
					<SelectTrigger id="type-select" aria-label="Training type">
						<SelectValue
							placeholder={
								family ? "Pick a training type…" : "Pick family first"
							}
						/>
					</SelectTrigger>
					<SelectContent>
						{types.isLoading && (
							<div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
								<Loader2 className="h-3 w-3 animate-spin" />
								Loading…
							</div>
						)}
						{types.data?.map((t) => (
							<SelectItem key={t.name} value={t.name}>
								{t.label}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
			</div>
		</div>
	);
}
