// Tiny inline sparkline — no axes, no tooltip. Just shape.

import { Line, LineChart, ResponsiveContainer } from "recharts";

interface Props {
	values: readonly number[];
	width?: number;
	height?: number;
	color?: string;
}

export function Sparkline({ values, width = 80, height = 24, color }: Props) {
	if (!values || values.length < 2) {
		return (
			<span
				className="inline-block bg-muted rounded"
				style={{ width, height: height / 2 }}
			/>
		);
	}
	const data = values.map((v, i) => ({ i, v }));
	return (
		<div style={{ width, height }} className="inline-block">
			<ResponsiveContainer width="100%" height="100%">
				<LineChart
					data={data}
					margin={{ top: 2, right: 2, bottom: 2, left: 2 }}
				>
					<Line
						type="monotone"
						dataKey="v"
						stroke={color ?? "var(--accent-bracket)"}
						strokeWidth={1.5}
						dot={false}
						isAnimationActive={false}
					/>
				</LineChart>
			</ResponsiveContainer>
		</div>
	);
}
