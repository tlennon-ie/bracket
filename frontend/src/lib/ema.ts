// Pure EMA. The smoothing slider feeds `alpha` ∈ [0, 0.99].
// alpha=0 → identity, alpha→1 → ultra-smooth (laggy).

export function applyEMA(raw: readonly number[], alpha: number): number[] {
	if (raw.length === 0) return [];
	const a = Math.max(0, Math.min(0.99, alpha));
	if (a === 0) return raw.slice();
	const out = new Array<number>(raw.length);
	let prev = raw[0] ?? 0;
	out[0] = prev;
	for (let i = 1; i < raw.length; i++) {
		const v = raw[i] ?? prev;
		prev = a * prev + (1 - a) * v;
		out[i] = prev;
	}
	return out;
}
