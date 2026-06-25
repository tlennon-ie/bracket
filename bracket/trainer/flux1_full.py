"""Flux.1 full fine-tune trainer — wraps sd-scripts/flux_train.py.

FLUX.1 (dev/schnell) full fine-tuning lives in **sd-scripts**, not musubi.
Mirrors :mod:`bracket.trainer.sdxl_full` (the sd-scripts full-FT pattern):
  - constructor takes ``sd_scripts_dir`` (not ``musubi_dir``)
  - sd-scripts dataset TOML via ``derive_run_toml(target_format="sd-scripts")``
  - INLINE latent caching (``--cache_latents --cache_latents_to_disk``)
  - samples land in ``<output>/sample`` (singular, sd-scripts default)

12B parameters means ``--blocks_to_swap`` is essential below 80 GB. The
canonical full-FT optimizer is Adafactor + ``--fused_backward_pass`` +
``--full_bf16`` with ``--max_grad_norm 0`` (fused backward doesn't compose
with gradient clipping). FLUX.1-specific flags (``--clip_l`` / ``--t5xxl`` /
``--ae`` / ``--guidance_scale`` / ``--timestep_sampling`` /
``--model_prediction_type``) come from sd-scripts/docs/flux_train_network.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.dataset.runtime import derive_run_toml
from bracket.hardware import detect_gpu, vram_tier
from bracket.search.space import (
    CategoricalKnob, FixedKnob, FloatKnob, IntKnob, SearchSpace,
)
from bracket.trainer.base import (
    LaunchSpec, Trainer, TrainerConfig,
    make_accelerate_launch_prefix, make_subprocess_env, resolve_save_every_n_steps,
)


_BATCH_BY_TIER = {
    "xl": (1, 2), "large": (1,), "high": (1,), "med": (1,), "low": (1,), "tiny": (1,),
}
_DEFAULT_BATCH_BY_TIER = {"xl": 2, "large": 1, "high": 1, "med": 1, "low": 1, "tiny": 1}
_FULL_SWAP_BY_TIER = {
    "xl": 0, "large": 12, "high": 24, "med": 32, "low": 38, "tiny": 38,
}


@dataclass
class Flux1FullConfig(TrainerConfig):
    learning_rate: float = 1e-6
    optimizer_type: str = "Adafactor"
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 50
    discrete_flow_shift: float = 3.0
    guidance_scale: float = 1.0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    full_bf16: bool = True
    fused_backward_pass: bool = True
    max_grad_norm: float = 0.0
    blocks_to_swap: int = 24


class Flux1FullTrainer(Trainer):
    name = "flux1-full-sd-scripts"

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
        if not (self.sd_scripts_dir / "flux_train.py").exists():
            raise FileNotFoundError(f"flux_train.py not found in {self.sd_scripts_dir}")
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(
            name=f"flux1-full-v0.1-{self.tier}",
            knobs={
                "learning_rate": FloatKnob(low=1e-7, high=1e-5, log=True),
                "optimizer_type": CategoricalKnob(choices=("Adafactor",)),
                "lr_scheduler": CategoricalKnob(
                    choices=("constant_with_warmup", "constant", "cosine"),
                ),
                "lr_warmup_steps": IntKnob(low=10, high=200),
                "discrete_flow_shift": FloatKnob(low=2.0, high=4.5),
                "train_batch_size": CategoricalKnob(choices=_BATCH_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "full_bf16": FixedKnob(value=True),
                "fused_backward_pass": FixedKnob(value=True),
                "max_grad_norm": FixedKnob(value=0.0),
                "guidance_scale": FixedKnob(value=1.0),
                "blocks_to_swap": FixedKnob(value=_FULL_SWAP_BY_TIER[self.tier]),
            },
        )

    def baseline_config(self) -> Flux1FullConfig:
        return Flux1FullConfig(
            train_batch_size=_DEFAULT_BATCH_BY_TIER[self.tier],
            blocks_to_swap=_FULL_SWAP_BY_TIER[self.tier],
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = _DEFAULT_BATCH_BY_TIER[self.tier]
        swap = _FULL_SWAP_BY_TIER[self.tier]
        return [
            Flux1FullConfig(
                learning_rate=5e-6, lr_scheduler="constant_with_warmup",
                lr_warmup_steps=50, discrete_flow_shift=3.0,
                train_batch_size=bs, blocks_to_swap=swap,
            ),
            Flux1FullConfig(
                learning_rate=1e-5, lr_scheduler="cosine",
                lr_warmup_steps=200, discrete_flow_shift=3.0,
                train_batch_size=bs, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> Flux1FullConfig:
        return Flux1FullConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer_type=str(knobs["optimizer_type"]),
            lr_scheduler=str(knobs["lr_scheduler"]),
            lr_warmup_steps=int(knobs["lr_warmup_steps"]),
            discrete_flow_shift=float(knobs["discrete_flow_shift"]),
            guidance_scale=float(knobs["guidance_scale"]),
            train_batch_size=int(knobs["train_batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            mixed_precision=str(knobs["mixed_precision"]),
            full_bf16=bool(knobs["full_bf16"]),
            fused_backward_pass=bool(knobs["fused_backward_pass"]),
            max_grad_norm=float(knobs["max_grad_norm"]),
            blocks_to_swap=int(knobs["blocks_to_swap"]),
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
        if not isinstance(config, Flux1FullConfig):
            raise TypeError(f"expected Flux1FullConfig, got {type(config).__name__}")
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "sample"  # sd-scripts default — singular
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
            "flux_train.py",
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
            "--gradient_checkpointing",
            "--cache_latents",
            "--cache_latents_to_disk",
            # Cache the (frozen) text-encoder outputs too, so CLIP-L + T5-XXL
            # don't stay resident in VRAM for the whole run — without this a
            # full FT of the 12B DiT OOMs on anything below ~40GB even with
            # --blocks_to_swap. Mirrors the LoRA adapter.
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--sdpa",
            "--save_precision", "bf16",
            "--save_model_as", "safetensors",
            "--save_every_n_steps", str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
        ]
        # Skip the per-image latent integrity check when already cached.
        from bracket.dataset.latent_cache import dataset_has_cached_latents
        if dataset_has_cached_latents(dataset_toml):
            cmd.append("--skip_cache_check")
        if config.full_bf16:
            cmd.append("--full_bf16")
        if config.fused_backward_pass:
            cmd.append("--fused_backward_pass")
        if config.optimizer_type == "Adafactor":
            cmd += [
                "--optimizer_args",
                "relative_step=False", "scale_parameter=False", "warmup_init=False",
            ]
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
