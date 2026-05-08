import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import type { FieldSpec } from "@/types/domain";
import { FileCode, Folder, Type as TypeIcon } from "lucide-react";
import type { ComponentType, SVGProps } from "react";

type IconType = ComponentType<SVGProps<SVGSVGElement>>;

function iconFor(kind: string): IconType {
	switch (kind) {
		case "dir":
			return Folder;
		case "path":
		case "filepath_optional":
			return FileCode;
		default:
			return TypeIcon;
	}
}

interface Props {
	fields: FieldSpec[];
	values: Record<string, string>;
	onChange: (name: string, value: string) => void;
	errors?: Record<string, string | undefined>;
}

export function DynamicFieldList({ fields, values, onChange, errors }: Props) {
	if (fields.length === 0) return null;
	return (
		<div className="grid grid-cols-1 gap-4">
			{fields.map((f) => {
				const Icon = iconFor(f.kind);
				const id = `field-${f.name}`;
				const err = errors?.[f.name];
				return (
					<div key={f.name} className="flex flex-col gap-1.5">
						<div className="flex items-center justify-between">
							<Label htmlFor={id} className="flex items-center gap-1.5">
								<Icon className="h-3 w-3 opacity-60" />
								<span>{f.label}</span>
								{f.required ? (
									<span className="text-primary">*</span>
								) : (
									<span className="text-[10px] text-muted-foreground normal-case tracking-normal">
										(optional)
									</span>
								)}
							</Label>
							{f.help && (
								<span
									className="text-[10px] text-muted-foreground max-w-[60%] truncate"
									title={f.help}
								>
									{f.help}
								</span>
							)}
						</div>
						<Input
							id={id}
							type="text"
							spellCheck={false}
							autoComplete="off"
							placeholder={f.default || " "}
							value={values[f.name] ?? f.default ?? ""}
							onChange={(e) => onChange(f.name, e.target.value)}
							className={cn(
								err && "border-destructive focus-visible:ring-destructive",
							)}
							aria-invalid={!!err}
							aria-describedby={err ? `${id}-err` : undefined}
						/>
						{err && (
							<p id={`${id}-err`} className="text-xs text-destructive">
								{err}
							</p>
						)}
					</div>
				);
			})}
		</div>
	);
}
