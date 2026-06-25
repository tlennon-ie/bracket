"""Flux.1 LoRA trainer adapter — wraps sd-scripts/flux_train_network.py.

FLUX.1 (dev/schnell) training lives in **sd-scripts**, not musubi-tuner.
It uses a Transformer-based DiT (~12B) with dual text encoders (T5-XXL +
CLIP-L) and a dedicated AutoEncoder (passed as ``--ae``, not ``--vae``).

Mirrors the sd-scripts pattern used by :mod:`bracket.trainer.sdxl`:
  - constructor takes ``sd_scripts_dir`` (not ``musubi_dir``)
  - the sd-scripts launcher prefix (``make_accelerate_launch_prefix``)
  - sd-scripts dataset TOML via ``derive_run_toml(target_format="sd-scripts")``
  - INLINE latent + TE caching (``--cache_latents --cache_latents_to_disk``)
    — there is NO separate pre-cache step
  - samples land in ``<output>/sample`` (singular, sd-scripts default)

FLUX.1-specific flags (see sd-scripts/docs/flux_train_network.md):
  --clip_l / --t5xxl / --ae      text encoders + autoencoder
  --network_module networks.lora_flux
  --guidance_scale 1.0           disables the distilled guidance for training
  --timestep_sampling shift      with --discrete_flow_shift to control the shift
  --model_prediction_type raw    recommended for FLUX.1
  --network_train_unet_only      TE outputs are cached & frozen
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.dataset.runtime import derive_run_toml
from bracket.hardware import (
    BLOCKS_TO_SWAP_BY_TIER, DATALOADER_WORKERS_BY_TIER,
    SDXL_LORA_BATCH_CHOICES_BY_TIER, SDXL_LORA_DEFAULT_BATCH_BY_TIER,
    detect_gpu, lora_grad_ckpt_baseline, lora_grad_ckpt_varies, vram_tier,
)
from bracket.search.space import (
    CategoricalKnob, FixedKnob, FloatKnob, IntKnob, SearchSpace,
)
from bracket.trainer.base import (
    LaunchSpec, Trainer, TrainerConfig,
    make_accelerate_launch_prefix, make_subprocess_env, resolve_save_every_n_steps,
)


@dataclass
class Flux1LoRAConfig(TrainerConfig):
    learning_rate: float = 1e-4
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 16.0
    discrete_flow_shift: float = 3.0
    guidance_scale: float = 1.0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    fp8_base: bool = True
    gradient_checkpointing: bool = True
    blocks_to_swap: int = 0
    dataloader_workers: int = 2


class Flux1LoRATrainer(Trainer):
    name = "flux1-lora-sd-scripts"

    def __init__(
        self,
        *,
        sd_scripts_dir: Path,
        venv_python: Path,
        dit_path: str,
        vae_path: str,
        t5xxl_path: str,
        clip_l_path: str,
        vram_gb: Optional[float] = None,
    ) -> None:
        self.sd_scripts_dir = Path(sd_scripts_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.t5xxl_path = t5xxl_path
        self.clip_l_path = clip_l_path
        if not (self.sd_scripts_dir / "flux_train_network.py").exists():
            raise FileNotFoundError(
                f"flux_train_network.py not found in {self.sd_scripts_dir}"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        if lora_grad_ckpt_varies(self.tier, "flux1"):
            ckpt_knob = CategoricalKnob(choices=(True, False))
        else:
            ckpt_knob = FixedKnob(value=True)
        return SearchSpace(
            name=f"flux1-lora-v0.1-{self.tier}",
            knobs={
                "learning_rate": FloatKnob(low=1e-6, high=2e-4, log=True),
                "optimizer_type": CategoricalKnob(
                    choices=("AdamW8bit", "AdamW", "Lion", "Prodigy", "Adafactor"),
                ),
                "lr_scheduler": CategoricalKnob(
                    choices=("cosine", "constant", "constant_with_warmup", "linear"),
                ),
                "lr_warmup_steps": IntKnob(low=0, high=200),
                "network_dim": CategoricalKnob(choices=(8, 16, 32, 64)),
                "network_alpha": CategoricalKnob(choices=(4.0, 8.0, 16.0, 32.0)),
                "discrete_flow_shift": FloatKnob(low=2.0, high=4.5),
                "train_batch_size": CategoricalKnob(
                    choices=SDXL_LORA_BATCH_CHOICES_BY_TIER[self.tier],
                ),
                "gradient_checkpointing": ckpt_knob,
                # Pinned.
                "dataloader_workers": FixedKnob(value=DATALOADER_WORKERS_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
                "guidance_scale": FixedKnob(value=1.0),
                "fp8_base": FixedKnob(value=True),
                "blocks_to_swap": FixedKnob(value=BLOCKS_TO_SWAP_BY_TIER[self.tier]),
            },
        )

    def baseline_config(self) -> Flux1LoRAConfig:
        return Flux1LoRAConfig(
            train_batch_size=SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, "flux1"),
            dataloader_workers=DATALOADER_WORKERS_BY_TIER[self.tier],
            blocks_to_swap=BLOCKS_TO_SWAP_BY_TIER[self.tier],
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, "flux1")
        workers = DATALOADER_WORKERS_BY_TIER[self.tier]
        swap = BLOCKS_TO_SWAP_BY_TIER[self.tier]
        return [
            # 1) ostris/Civitai-style starter — 5e-5 + flow_shift=3.
            Flux1LoRAConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 2) Prodigy.
            Flux1LoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 3) Higher dim — character LoRAs benefit from rank 64 on Flux.1.
            Flux1LoRAConfig(
                learning_rate=1e-4, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=64, network_alpha=32.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> Flux1LoRAConfig:
        return Flux1LoRAConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer_type=str(knobs["optimizer_type"]),
            lr_scheduler=str(knobs["lr_scheduler"]),
            lr_warmup_steps=int(knobs["lr_warmup_steps"]),
            network_dim=int(knobs["network_dim"]),
            network_alpha=float(knobs["network_alpha"]),
            discrete_flow_shift=float(knobs["discrete_flow_shift"]),
            guidance_scale=float(knobs["guidance_scale"]),
            train_batch_size=int(knobs["train_batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            mixed_precision=str(knobs["mixed_precision"]),
            max_grad_norm=float(knobs["max_grad_norm"]),
            fp8_base=bool(knobs["fp8_base"]),
            gradient_checkpointing=bool(knobs["gradient_checkpointing"]),
            blocks_to_swap=int(knobs["blocks_to_swap"]),
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
        if not isinstance(config, Flux1LoRAConfig):
            raise TypeError(f"expected Flux1LoRAConfig, got {type(config).__name__}")
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        # sd-scripts writes samples to <output_dir>/sample (singular).
        sample_dir = output_dir / "sample"
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        run_toml = derive_run_toml(
            source_toml=dataset_toml,
            target_path=run_dir / "dataset.toml",
            batch_size=config.train_batch_size,
            target_format="sd-scripts",
        )

        cmd: list[str] = [
            *make_accelerate_launch_prefix(self.venv_python, mixed_precision=config.mixed_precision),
            "flux_train_network.py",
            "--pretrained_model_name_or_path", self.dit_path,
            "--clip_l", self.clip_l_path,
            "--t5xxl", self.t5xxl_path,
            "--ae", self.vae_path,
            "--dataset_config", str(run_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--network_module", "networks.lora_flux",
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
            # FLUX.1-specific training parameters.
            "--guidance_scale", f"{config.guidance_scale:.4g}",
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", f"{config.discrete_flow_shift:.4f}",
            "--model_prediction_type", "raw",
            # Inline caching — TE outputs are frozen, so train the DiT only.
            "--cache_latents",
            "--cache_latents_to_disk",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--network_train_unet_only",
            "--sdpa",
            "--save_precision", "bf16",
            "--save_model_as", "safetensors",
            "--no_metadata",
            "--save_every_n_steps", str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),
            "--max_data_loader_n_workers", str(config.dataloader_workers),
            "--persistent_data_loader_workers",
        ]
        # Skip the per-image latent integrity check when already cached.
        from bracket.dataset.latent_cache import dataset_has_cached_latents
        if dataset_has_cached_latents(dataset_toml):
            cmd.append("--skip_cache_check")
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if config.fp8_base:
            cmd.append("--fp8_base")
        if config.blocks_to_swap > 0:
            cmd += ["--blocks_to_swap", str(config.blocks_to_swap)]
        if sample_prompts is not None and sample_every_n_steps:
            cmd += [
                "--sample_prompts", str(sample_prompts),
                "--sample_every_n_steps", str(sample_every_n_steps),
            ]
        if save_state:
            cmd.append("--save_state")
        if resume_from is not None:
            cmd += ["--resume", str(resume_from)]

        return LaunchSpec(
            cmd=cmd, cwd=self.sd_scripts_dir, env=make_subprocess_env(),
            output_dir=output_dir, logging_dir=logging_dir,
            tfevents_glob=str(logging_dir / "**" / "events.out.tfevents.*"),
            sample_dir=sample_dir,
        )
