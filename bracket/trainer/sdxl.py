"""SDXL LoRA trainer adapter for sd-scripts/sdxl_train_network.py.

The orchestrator picks a config (sampled from declare_search_space()), calls
prepare_run(), and gets a LaunchSpec it can hand to the runner. We compose
the CLI as a list of strings — never a shell string — so quoting doesn't bite.

Defaults match sd-scripts' documented sane SDXL LoRA setup so the baseline
config corresponds to "what a human running sd-scripts by hand would type."
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.trainer.base import (
    LaunchSpec, Trainer, TrainerConfig,
    make_accelerate_launch_prefix, make_subprocess_env, resolve_save_every_n_steps,
)
from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchSpace,
)
from bracket.hardware import (
    DATALOADER_WORKERS_BY_TIER,
    SDXL_LORA_BATCH_CHOICES_BY_TIER,
    SDXL_LORA_DEFAULT_BATCH_BY_TIER,
    detect_gpu,
    lora_grad_ckpt_baseline,
    lora_grad_ckpt_varies,
    vram_tier,
)


# Where the user's HF cache lives — passed in by CLI but a sane default for
# the smoke test. Override with --pretrained-model on the orchestrator CLI.
DEFAULT_SDXL_PRETRAINED = (
    "C:/Users/compu/.cache/huggingface/hub/"
    "models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/"
    "462165984030d82259a11f4367a4eed129e94a7b"
)


@dataclass
class SDXLConfig(TrainerConfig):
    learning_rate: float = 1e-4
    unet_lr: Optional[float] = None
    text_encoder_lr: Optional[float] = None
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 16.0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    noise_offset: float = 0.0
    min_snr_gamma: Optional[float] = None
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    dataloader_workers: int = 2


class SDXLTrainer(Trainer):
    name = "sdxl-sd-scripts"

    def __init__(
        self,
        *,
        sd_scripts_dir: Path,
        venv_python: Path,
        pretrained_model: str = DEFAULT_SDXL_PRETRAINED,
        vram_gb: Optional[float] = None,
    ) -> None:
        self.sd_scripts_dir = Path(sd_scripts_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.pretrained_model = pretrained_model
        if not (self.sd_scripts_dir / "sdxl_train_network.py").exists():
            raise FileNotFoundError(
                f"sdxl_train_network.py not found in {self.sd_scripts_dir}"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")

        # Resolve VRAM either from explicit caller arg, GPU detection, or fallback.
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info is not None else 12.0
        self.vram_gb: float = float(vram_gb)
        self.tier: str = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        batch_choices = SDXL_LORA_BATCH_CHOICES_BY_TIER[self.tier]
        # Variable gradient checkpointing on high-VRAM tiers (off = ~25% faster);
        # pinned ON when the model+batch+VRAM combo can't fit without it.
        if lora_grad_ckpt_varies(self.tier, "sdxl"):
            ckpt_knob = CategoricalKnob(choices=(True, False))
        else:
            ckpt_knob = FixedKnob(value=True)
        return SearchSpace(
            name=f"sdxl-lora-v0.3-{self.tier}",
            knobs={
                # Loss-bearing knobs.
                "learning_rate": FloatKnob(low=1e-6, high=5e-4, log=True),
                "optimizer_type": CategoricalKnob(
                    choices=("AdamW8bit", "AdamW", "Lion", "Prodigy", "Adafactor")
                ),
                "lr_scheduler": CategoricalKnob(
                    choices=("cosine", "constant", "constant_with_warmup", "linear")
                ),
                "lr_warmup_steps": IntKnob(low=0, high=200),
                "network_dim": CategoricalKnob(choices=(8, 16, 32, 64)),
                "network_alpha": CategoricalKnob(choices=(4.0, 8.0, 16.0, 32.0)),
                "noise_offset": FloatKnob(low=0.0, high=0.1),
                # Throughput knobs.
                "train_batch_size": CategoricalKnob(choices=batch_choices),
                "gradient_checkpointing": ckpt_knob,
                # Pinned.
                "dataloader_workers": FixedKnob(value=DATALOADER_WORKERS_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
            },
        )

    def baseline_config(self) -> SDXLConfig:
        # Tier-appropriate defaults: batch_size + gradient_checkpointing both
        # respect the detected VRAM. On a 32GB 5090, ckpt defaults to False —
        # the orchestrator can search True if a candidate needs the headroom.
        return SDXLConfig(
            learning_rate=1e-4,
            optimizer_type="AdamW8bit",
            lr_scheduler="cosine",
            lr_warmup_steps=100,
            network_dim=32,
            network_alpha=16.0,
            train_batch_size=SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_accumulation_steps=1,
            mixed_precision="bf16",
            noise_offset=0.0,
            max_grad_norm=1.0,
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, "sdxl"),
            dataloader_workers=DATALOADER_WORKERS_BY_TIER[self.tier],
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, "sdxl")
        workers = DATALOADER_WORKERS_BY_TIER[self.tier]
        return [
            # 1) Kohya recommended starter — slightly lower LR with warmup.
            SDXLConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
            # 2) Prodigy auto-LR — community-favoured for "just works" runs.
            SDXLConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=16.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
            # 3) Smaller dim/alpha — for small / homogeneous datasets where
            #    32 overfits.
            SDXLConfig(
                learning_rate=1e-4, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=16, network_alpha=8.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> SDXLConfig:
        return SDXLConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer_type=str(knobs["optimizer_type"]),
            lr_scheduler=str(knobs["lr_scheduler"]),
            lr_warmup_steps=int(knobs["lr_warmup_steps"]),
            network_dim=int(knobs["network_dim"]),
            network_alpha=float(knobs["network_alpha"]),
            train_batch_size=int(knobs["train_batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            mixed_precision=str(knobs["mixed_precision"]),
            noise_offset=float(knobs["noise_offset"]),
            max_grad_norm=float(knobs["max_grad_norm"]),
            gradient_checkpointing=bool(knobs["gradient_checkpointing"]),
            dataloader_workers=int(knobs["dataloader_workers"]),
        )

    def prepare_run(
        self,
        *,
        run_dir: Path,
        config: TrainerConfig,
        dataset_toml: Path,
        max_steps: int,
        seed: int,
        sample_prompts: Optional[Path] = None,
        sample_every_n_steps: Optional[int] = None,
        save_every_n_steps: Optional[int] = None,
        save_state: bool = False,
        resume_from: Optional[Path] = None,
    ) -> LaunchSpec:
        if not isinstance(config, SDXLConfig):
            raise TypeError(f"expected SDXLConfig, got {type(config).__name__}")

        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        # sd-scripts writes samples to <output_dir>/sample (singular) — not "samples"
        sample_dir = output_dir / "sample"
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            *make_accelerate_launch_prefix(self.venv_python, mixed_precision=config.mixed_precision),
            "sdxl_train_network.py",
            "--pretrained_model_name_or_path", self.pretrained_model,
            "--dataset_config", str(dataset_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--network_module", "networks.lora",
            "--network_dim", str(config.network_dim),
            "--network_alpha", str(config.network_alpha),
            "--learning_rate", f"{config.learning_rate:.10g}",
            "--optimizer_type", config.optimizer_type,
            "--lr_scheduler", config.lr_scheduler,
            "--lr_warmup_steps", str(config.lr_warmup_steps),
            "--train_batch_size", str(config.train_batch_size),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--mixed_precision", config.mixed_precision,
            "--max_grad_norm", str(config.max_grad_norm),
            "--max_train_steps", str(max_steps),
            "--seed", str(seed),
            "--cache_latents",
            "--cache_latents_to_disk",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            # Required when caching TE outputs: TE weights stay frozen and
            # cached outputs are reused. Standard SDXL LoRA pattern.
            "--network_train_unet_only",
            "--sdpa",
            "--save_precision", "bf16",
            "--save_model_as", "safetensors",
            # No checkpoints written mid-run — orchestrator scores from tfevents,
            # we don't keep weights for losers.
            "--no_metadata",
            "--save_every_n_steps", str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),  # only at end
            "--max_data_loader_n_workers", str(config.dataloader_workers),
            "--persistent_data_loader_workers",
        ]
        # Skip the per-image latent integrity check when the dataset is
        # already cached (saves several minutes on large sets).
        from bracket.dataset.latent_cache import dataset_has_cached_latents
        if dataset_has_cached_latents(dataset_toml):
            cmd.append("--skip_cache_check")
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")

        if config.noise_offset and config.noise_offset > 0:
            cmd += ["--noise_offset", f"{config.noise_offset:.6f}"]
        if config.min_snr_gamma is not None:
            cmd += ["--min_snr_gamma", str(config.min_snr_gamma)]
        if config.unet_lr is not None:
            cmd += ["--unet_lr", f"{config.unet_lr:.10g}"]
        if config.text_encoder_lr is not None:
            cmd += ["--text_encoder_lr", f"{config.text_encoder_lr:.10g}"]
        if sample_prompts is not None and sample_every_n_steps:
            cmd += [
                "--sample_prompts", str(sample_prompts),
                "--sample_every_n_steps", str(sample_every_n_steps),
                "--sample_sampler", "euler_a",
            ]

        if save_state:
            cmd.append("--save_state")
        if resume_from is not None:
            cmd += ["--resume", str(resume_from)]

        return LaunchSpec(
            cmd=cmd,
            cwd=self.sd_scripts_dir,
            env=make_subprocess_env(),
            output_dir=output_dir,
            logging_dir=logging_dir,
            tfevents_glob=str(logging_dir / "**" / "events.out.tfevents.*"),
            sample_dir=sample_dir,
        )
