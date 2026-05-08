// Largest-Triangle-Three-Buckets downsampling.
// Keeps visual fidelity for line charts at a fraction of the points.
//
// Reference: Steinarsson, "Downsampling Time Series for Visual Representation" (2013).

export interface Point {
	x: number;
	y: number;
}

const ZERO: Point = { x: 0, y: 0 };

function at(arr: readonly Point[], i: number): Point {
	return arr[i] ?? ZERO;
}

export function lttb(data: readonly Point[], threshold: number): Point[] {
	const n = data.length;
	if (threshold >= n || threshold <= 2) {
		return data.slice();
	}
	const sampled: Point[] = [];
	const bucketSize = (n - 2) / (threshold - 2);
	let a = 0;
	sampled.push(at(data, 0));

	for (let i = 0; i < threshold - 2; i++) {
		const avgRangeStart = Math.floor((i + 1) * bucketSize) + 1;
		const avgRangeEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, n);
		const avgRangeLen = Math.max(1, avgRangeEnd - avgRangeStart);
		let avgX = 0;
		let avgY = 0;
		for (let k = avgRangeStart; k < avgRangeEnd; k++) {
			const p = at(data, k);
			avgX += p.x;
			avgY += p.y;
		}
		avgX /= avgRangeLen;
		avgY /= avgRangeLen;

		const rangeStart = Math.floor(i * bucketSize) + 1;
		const rangeEnd = Math.floor((i + 1) * bucketSize) + 1;
		const pointA = at(data, a);
		let maxArea = -1;
		let nextA = rangeStart;
		let chosen: Point = at(data, rangeStart);
		for (let k = rangeStart; k < rangeEnd; k++) {
			const p = at(data, k);
			const area = Math.abs(
				(pointA.x - avgX) * (p.y - pointA.y) -
					(pointA.x - p.x) * (avgY - pointA.y),
			);
			if (area > maxArea) {
				maxArea = area;
				chosen = p;
				nextA = k;
			}
		}
		sampled.push(chosen);
		a = nextA;
	}
	sampled.push(at(data, n - 1));
	return sampled;
}
