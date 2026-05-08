// Lightweight keyboard-shortcut hook. No deps, ~30 LOC.
//
// Supports single-key shortcuts and the `g <letter>` chord pattern (Vim-ish
// "go to ..." nav). Ignores keypresses while focused on form inputs.

import { useEffect } from "react";

export interface Shortcut {
	/** Key like 'r', 'Escape', '?'. Comparison is case-sensitive vs key.toLowerCase(). */
	key: string;
	/** Optional first key in a chord — e.g. {chord: 'g', key: 's'} for `g s`. */
	chord?: string;
	ctrl?: boolean;
	meta?: boolean;
	shift?: boolean;
	handler: (e: KeyboardEvent) => void;
	/** If true, the shortcut still fires when typing in an input. */
	whenFocused?: boolean;
}

const FORM_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);

function isTypingTarget(target: EventTarget | null): boolean {
	if (!(target instanceof HTMLElement)) return false;
	if (FORM_TAGS.has(target.tagName)) return true;
	if (target.isContentEditable) return true;
	return false;
}

export function useKeyboardShortcuts(shortcuts: Shortcut[]): void {
	useEffect(() => {
		let chordPending: string | null = null;
		let chordTimer: ReturnType<typeof setTimeout> | null = null;

		const onKeyDown = (e: KeyboardEvent) => {
			const k = e.key;
			const lk = k.toLowerCase();
			const inForm = isTypingTarget(e.target);

			// Chord step 1: capture chord prefix.
			const chordCandidates = shortcuts.filter(
				(s) => s.chord && s.chord.toLowerCase() === lk,
			);
			if (
				chordCandidates.length > 0 &&
				!chordPending &&
				!e.ctrlKey &&
				!e.metaKey &&
				!inForm
			) {
				chordPending = lk;
				if (chordTimer) clearTimeout(chordTimer);
				chordTimer = setTimeout(() => {
					chordPending = null;
				}, 1200);
				return;
			}

			for (const s of shortcuts) {
				if (s.chord) {
					if (chordPending !== s.chord.toLowerCase()) continue;
					if (s.key.toLowerCase() !== lk) continue;
					chordPending = null;
					if (chordTimer) clearTimeout(chordTimer);
					if (!s.whenFocused && inForm) continue;
					e.preventDefault();
					s.handler(e);
					return;
				}
				if (s.key.toLowerCase() !== lk) continue;
				if (!!s.ctrl !== e.ctrlKey) continue;
				if (!!s.meta !== e.metaKey) continue;
				if (!!s.shift !== e.shiftKey) continue;
				if (!s.whenFocused && inForm) continue;
				e.preventDefault();
				s.handler(e);
				return;
			}
		};

		window.addEventListener("keydown", onKeyDown);
		return () => {
			window.removeEventListener("keydown", onKeyDown);
			if (chordTimer) clearTimeout(chordTimer);
		};
	}, [shortcuts]);
}
