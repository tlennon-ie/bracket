"""Diagnostic: re-judge every sample image in a completed run and report
which ones failed and why.

Usage:
    python scripts/rejudge_run.py <run_dir> <prompts_file> [--model NAME]

Example:
    python scripts/rejudge_run.py \\
        ./runs/portraits-001/runs/baseline-000-s0-1778245819 \\
        ./configs/sample_prompts.txt

The script connects to LMStudio at http://localhost:1234/v1, sends every
PNG in <run_dir>/output/sample/ paired with the prompt that produced it,
and prints a table showing pass/fail for each image plus the error
message for failures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bracket.judge.base import parse_judge_prompts_file
from bracket.judge.lmstudio import LMStudioJudge, LMStudioJudgeConfig
from bracket.orchestrator.scorer import _pair_samples_with_prompts


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-judge a completed run's samples")
    ap.add_argument("run_dir", type=Path, help="path to the run directory (contains output/sample/)")
    ap.add_argument("prompts_file", type=Path, help="sample_prompts.txt that drove sampling")
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="qwen3-vl-8b-thinking-abliterated")
    args = ap.parse_args()

    sample_dir = args.run_dir / "output" / "sample"
    if not sample_dir.exists():
        print(f"!! sample dir not found: {sample_dir}")
        return 1
    prompts = parse_judge_prompts_file(args.prompts_file)
    if not prompts:
        print(f"!! no prompts parsed from {args.prompts_file}")
        return 1
    pairing = _pair_samples_with_prompts(sample_dir, prompts)
    if not pairing:
        print(f"!! pairing produced no matches (filename format issue?). "
              f"Saw {len(list(sample_dir.glob('*.png')))} PNGs and {len(prompts)} prompts.")
        return 1

    judge = LMStudioJudge(LMStudioJudgeConfig(base_url=args.base_url, model=args.model))
    print(f"Judging {len(pairing)} images via {args.model} at {args.base_url}\n")

    n_ok = n_fail = 0
    width = max(len(p.name) for p in pairing) if pairing else 40
    for img, prompt in sorted(pairing.items()):
        j = judge.judge_image(img, prompt)
        if j.error:
            n_fail += 1
            print(f"FAIL  {img.name:<{width}}  {j.error}")
            print(f"      prompt: {prompt[:80]}…")
            if j.raw_response:
                snippet = j.raw_response.replace("\n", " ")[:200]
                print(f"      raw:    {snippet}")
        else:
            n_ok += 1
            print(f"OK    {img.name:<{width}}  "
                  f"PA={j.prompt_adherence:.1f} VQ={j.visual_quality:.1f} "
                  f"AF={j.artifact_free:.1f} → {j.overall:.2f}")

    print(f"\nDone: {n_ok} ok, {n_fail} failed.")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
