// Per-run accordion. Lazy-loaded thumbs, lightbox on click.

import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { shortenRunId } from "@/lib/format";
import type { GalleryGroup, GalleryItem } from "@/types/domain";
import { Image as ImageIcon } from "lucide-react";
import { useMemo, useState } from "react";

interface Props {
	groups: GalleryGroup[];
}

export function GalleryAccordion({ groups }: Props) {
	const [lightbox, setLightbox] = useState<GalleryItem | null>(null);

	const sorted = useMemo(
		() => [...groups].sort((a, b) => b.mtime - a.mtime),
		[groups],
	);

	if (sorted.length === 0) {
		return (
			<div className="text-sm text-muted-foreground py-8 text-center border border-dashed border-border rounded-lg">
				No samples yet. They'll appear once{" "}
				<code className="font-mono-tight">--sample_every_n_steps</code> fires.
			</div>
		);
	}

	return (
		<>
			<Accordion
				type="multiple"
				defaultValue={sorted.slice(0, 1).map((g) => g.run_id)}
				className="w-full"
			>
				{sorted.map((g) => (
					<AccordionItem key={g.run_id} value={g.run_id}>
						<AccordionTrigger>
							<div className="flex items-center gap-3">
								<ImageIcon className="h-4 w-4 text-muted-foreground" />
								<code className="font-mono-tight text-sm">
									{shortenRunId(g.run_id, 30)}
								</code>
								<Badge variant="muted" className="text-[10px]">
									{g.items.length} sample{g.items.length === 1 ? "" : "s"}
								</Badge>
							</div>
						</AccordionTrigger>
						<AccordionContent>
							<div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
								{g.items.slice(0, 24).map((item) => (
									<button
										key={item.url}
										type="button"
										onClick={() => setLightbox(item)}
										className="group relative aspect-square overflow-hidden rounded-md border border-border bg-muted hover:border-primary transition-colors"
										aria-label={`Open ${item.caption}`}
									>
										<img
											loading="lazy"
											src={item.url}
											alt={item.caption}
											className="absolute inset-0 h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.02]"
										/>
										<span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-2 text-[10px] text-white truncate font-mono-tight opacity-0 group-hover:opacity-100 transition-opacity">
											{item.caption}
										</span>
									</button>
								))}
								{g.items.length > 24 && (
									<div className="col-span-full text-xs text-muted-foreground text-center py-2">
										{g.items.length - 24} more samples in this run
									</div>
								)}
							</div>
						</AccordionContent>
					</AccordionItem>
				))}
			</Accordion>

			<Dialog
				open={!!lightbox}
				onOpenChange={(open) => !open && setLightbox(null)}
			>
				<DialogContent className="max-w-3xl">
					{lightbox && (
						<div className="flex flex-col gap-3">
							<div className="flex items-center gap-2 text-xs text-muted-foreground">
								<code className="font-mono-tight">{lightbox.run_id}</code>
								<span>·</span>
								<span>{lightbox.caption}</span>
							</div>
							<img
								src={lightbox.url}
								alt={lightbox.caption}
								className="w-full h-auto rounded-md"
							/>
						</div>
					)}
				</DialogContent>
			</Dialog>
		</>
	);
}
