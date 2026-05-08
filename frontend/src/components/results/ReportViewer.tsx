import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
	markdown: string | null;
	className?: string;
}

export function ReportViewer({ markdown, className }: Props) {
	if (!markdown) {
		return (
			<div className="text-sm text-muted-foreground py-12 text-center border border-dashed border-border rounded-lg">
				No report yet. Sessions generate{" "}
				<code className="font-mono-tight">report.md</code> on completion, or hit{" "}
				<span className="font-medium">Regenerate</span> to build one now.
			</div>
		);
	}
	return (
		<article
			className={cn(
				"prose-sm max-w-none text-sm leading-relaxed",
				"[&_h1]:text-2xl [&_h1]:font-semibold [&_h1]:mt-8 [&_h1]:mb-4 [&_h1]:font-display",
				"[&_h2]:text-lg [&_h2]:font-semibold [&_h2]:mt-6 [&_h2]:mb-3 [&_h2]:font-display [&_h2]:tracking-tight",
				"[&_h3]:text-base [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-2",
				"[&_p]:my-3 [&_p]:text-foreground/90",
				"[&_ul]:my-3 [&_ul]:pl-5 [&_ul]:list-disc",
				"[&_ol]:my-3 [&_ol]:pl-5 [&_ol]:list-decimal",
				"[&_li]:my-1",
				"[&_code]:font-mono-tight [&_code]:text-[0.9em] [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:bg-muted",
				"[&_pre]:bg-muted [&_pre]:rounded-md [&_pre]:p-4 [&_pre]:overflow-x-auto [&_pre]:text-xs",
				"[&_pre_code]:bg-transparent [&_pre_code]:p-0",
				"[&_table]:my-4 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs",
				"[&_th]:text-left [&_th]:font-semibold [&_th]:py-2 [&_th]:pr-3 [&_th]:border-b [&_th]:border-border",
				"[&_td]:py-2 [&_td]:pr-3 [&_td]:border-b [&_td]:border-border/60 [&_td]:font-mono-tight",
				"[&_a]:text-primary [&_a]:underline-offset-4 hover:[&_a]:underline",
				"[&_hr]:my-8 [&_hr]:border-border",
				"[&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-4 [&_blockquote]:italic [&_blockquote]:text-muted-foreground",
				className,
			)}
		>
			<ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
		</article>
	);
}
