"""bracket-orchestrate — autonomous AutoML loop for diffusion training.

Picks a trainer per --trainer (sdxl-lora | sdxl-full | zimage-lora | zimage-full),
samples configs from its search space, runs each, scores with optional VLM judge,
and writes a Markdown report with confidence intervals when --seeds-per-config>=2.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bracket.dataset.subset import DatasetSubsetSpec, build_subset
from bracket.judge.lmstudio import LMStudioJudge, LMStudioJudgeConfig
from bracket.orchestrator.finals import run_finals_stage
from bracket.orchestrator.loop import orchestrate
from bracket.proof.report import generate_report
from bracket.search.controller import RandomSearch, SearchController
from bracket.search.optuna_search import OptunaTPESearch
from bracket.trainer.base import Trainer
from bracket.trainer.sdxl import SDXLTrainer
from bracket.trainer.sdxl_full import SDXLFullTrainer
from bracket.trainer.zimage_full import ZImageFullTrainer
from bracket.trainer.zimage_lora import ZImageLoRATrainer


TRAINER_CHOICES = ("sdxl-lora", "sdxl-full", "zimage-lora", "zimage-full")


def _build_trainer(args: argparse.Namespace) -> Trainer:
    if args.trainer == "sdxl-lora":
        if not args.pretrained_model:
            raise SystemExit("--pretrained-model required for sdxl-lora")
        return SDXLTrainer(
            sd_scripts_dir=args.sd_scripts, venv_python=args.venv_python,
            pretrained_model=args.pretrained_model, vram_gb=args.vram_gb,
        )
    if args.trainer == "sdxl-full":
        if not args.pretrained_model:
            raise SystemExit("--pretrained-model required for sdxl-full")
        return SDXLFullTrainer(
            sd_scripts_dir=args.sd_scripts, venv_python=args.venv_python,
            pretrained_model=args.pretrained_model, vram_gb=args.vram_gb,
        )
    if args.trainer in ("zimage-lora", "zimage-full"):
        missing = [n for n, v in [
            ("--musubi", args.musubi),
            ("--dit", args.dit),
            ("--vae", args.vae),
            ("--text-encoder", args.text_encoder),
        ] if not v]
        if missing:
            raise SystemExit(f"missing required args for {args.trainer}: {missing}")
        cls = ZImageLoRATrainer if args.trainer == "zimage-lora" else ZImageFullTrainer
        return cls(
            musubi_dir=args.musubi, venv_python=args.venv_python,
            dit_path=args.dit, vae_path=args.vae, text_encoder_path=args.text_encoder,
            vram_gb=args.vram_gb,
        )
    raise SystemExit(f"unknown trainer {args.trainer!r}")


def main() -> int:
    p = argparse.ArgumentParser(prog="bracket-orchestrate")
    p.add_argument("--trainer", choices=TRAINER_CHOICES, default="sdxl-lora",
                   help="which model+mode to tune")

    # Trainer common
    p.add_argument("--venv-python", type=Path, required=True,
                   help="python.exe inside the trainer venv")
    p.add_argument("--vram-gb", type=float, default=None,
                   help="VRAM in GB; if omitted, auto-detected from CUDA")

    # sd-scripts trainers
    p.add_argument("--sd-scripts", type=Path,
                   help="path to sd-scripts checkout (sdxl-lora, sdxl-full)")
    p.add_argument("--pretrained-model", type=str,
                   help="HF dir or .safetensors for SDXL base (sdxl-lora, sdxl-full)")
    # musubi trainers
    p.add_argument("--musubi", type=Path,
                   help="path to musubi-tuner checkout (zimage-lora, zimage-full)")
    p.add_argument("--dit", type=str, help="path to DiT weights (zimage-*)")
    p.add_argument("--vae", type=str, help="path to VAE weights (zimage-*)")
    p.add_argument("--text-encoder", type=str, help="path to text encoder (zimage-*)")

    # Run config
    p.add_argument("--dataset-toml", type=Path, required=True,
                   help="source dataset TOML — flat-musubi or sd-scripts format")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--budget", type=int, default=8,
                   help="number of unique candidate configs to try (excl. baseline)")
    p.add_argument("--max-steps-per-run", type=int, default=200)
    p.add_argument("--max-wall-seconds-per-run", type=int, default=1200)
    p.add_argument("--images-per-dataset", type=int, default=16)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--search-seed", type=int, default=0)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--mirror-stdout", action="store_true",
                   help="echo each candidate's stdout to console as well as logfile")
    p.add_argument("--seeds-per-config", type=int, default=1,
                   help="run each config N times with different seeds; "
                        ">=2 enables confidence intervals in the report")
    p.add_argument("--search", choices=("random", "optuna"), default="optuna",
                   help="search controller: optuna (TPE, smart) or random (uniform)")
    p.add_argument("--optuna-startup-trials", type=int, default=5,
                   help="number of random trials before TPE kicks in")
    p.add_argument("--finals-top-k", type=int, default=0,
                   help="if >0, after stage 1 re-run the top-K configs at "
                        "--finals-max-steps to verify the short-run ranking")
    p.add_argument("--finals-max-steps", type=int, default=2000)
    p.add_argument("--finals-wall-seconds", type=int, default=7200)
    p.add_argument("--finals-seeds-per-config", type=int, default=2)

    # Sample generation
    p.add_argument("--sample-prompts", type=Path, default=None)
    p.add_argument("--sample-every-n-steps", type=int, default=None,
                   help="default: max_steps_per_run (samples once at end)")

    # VLM judge
    p.add_argument("--judge", choices=("none", "lmstudio"), default="none")
    p.add_argument("--judge-base-url", default="http://localhost:1234/v1")
    p.add_argument("--judge-model", default="qwen3-vl-8b-thinking-abliterated")
    p.add_argument("--judge-loss-weight", type=float, default=0.3)
    p.add_argument("--judge-sample-weight", type=float, default=0.7)

    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("orchestrate")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_dir = output_dir / "subset"
    subset_toml = build_subset(
        source_toml=args.dataset_toml,
        target_dir=subset_dir,
        spec=DatasetSubsetSpec(images_per_dataset=args.images_per_dataset, seed=args.subset_seed),
    )
    log.info("dataset subset written to %s", subset_toml)

    trainer = _build_trainer(args)
    space = trainer.declare_search_space()
    baseline = trainer.baseline_config()
    bs_choice = space.knobs.get("train_batch_size")
    log.info(
        "trainer=%s tier=%s vram=%.1f GB → batch_size choices=%s, baseline batch_size=%d",
        trainer.name, getattr(trainer, "tier", "?"), getattr(trainer, "vram_gb", 0.0),
        getattr(bs_choice, "choices", None) if bs_choice else None,
        getattr(baseline, "train_batch_size", 1),
    )

    sample_every = args.sample_every_n_steps
    if args.sample_prompts is not None and sample_every is None:
        sample_every = args.max_steps_per_run

    sample_judge = None
    if args.judge == "lmstudio":
        if args.sample_prompts is None:
            log.warning("--judge lmstudio set but no --sample-prompts; judge disabled")
        else:
            sample_judge = LMStudioJudge(LMStudioJudgeConfig(
                base_url=args.judge_base_url, model=args.judge_model,
            ))
            log.info("VLM judge: LMStudio model=%s base_url=%s loss_w=%.2f sample_w=%.2f",
                     args.judge_model, args.judge_base_url,
                     args.judge_loss_weight, args.judge_sample_weight)

    controller: SearchController
    if args.search == "optuna":
        controller = OptunaTPESearch(
            seed=args.search_seed,
            n_startup_trials=args.optuna_startup_trials,
        )
    else:
        controller = RandomSearch(seed=args.search_seed)
    log.info("search controller: %s", type(controller).__name__)
    result = orchestrate(
        trainer=trainer,
        dataset_toml=subset_toml,
        output_dir=output_dir,
        controller=controller,
        budget_runs=args.budget,
        max_steps_per_run=args.max_steps_per_run,
        max_wall_seconds_per_run=args.max_wall_seconds_per_run,
        seeds_per_config=args.seeds_per_config,
        skip_baseline=args.skip_baseline,
        mirror_stdout=args.mirror_stdout,
        sample_prompts=args.sample_prompts,
        sample_every_n_steps=sample_every,
        sample_judge=sample_judge,
        loss_weight=args.judge_loss_weight,
        sample_weight=args.judge_sample_weight,
    )

    if args.finals_top_k > 0:
        log.info("running finals stage: top %d → %d steps each, %d seeds",
                 args.finals_top_k, args.finals_max_steps, args.finals_seeds_per_config)
        finals = run_finals_stage(
            trainer=trainer, stage1=result, dataset_toml=subset_toml,
            output_dir=output_dir, top_k=args.finals_top_k,
            finals_max_steps=args.finals_max_steps,
            finals_max_wall_seconds=args.finals_wall_seconds,
            finals_seeds_per_config=args.finals_seeds_per_config,
            sample_prompts=args.sample_prompts,
            sample_every_n_steps=sample_every,
            sample_judge=sample_judge,
            loss_weight=args.judge_loss_weight,
            sample_weight=args.judge_sample_weight,
            mirror_stdout=args.mirror_stdout,
        )
        if finals.best_finalist is not None:
            log.info("finals best: %s score=%s",
                     finals.best_finalist.run_id, finals.best_finalist.score)

    report_path = generate_report(result.ledger_path, output_dir / "report.md")
    log.info("orchestration complete: %d runs, %d disqualified",
             result.n_completed, result.n_disqualified)
    if result.best_entry:
        log.info("best run: %s score=%s", result.best_entry.run_id, result.best_entry.score)
    log.info("report written to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
