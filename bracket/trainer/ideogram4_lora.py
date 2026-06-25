"""Ideogram 4 LoRA trainer adapter — wraps musubi-tuner ideogram4_train_network.

Ideogram 4 (musubi docs/ideogram4.md) is a text-to-image model: a 34-layer
single-stream transformer DiT, conditioned by a Qwen3-VL-8B text encoder, with
the Flux2 KL-VAE. This adapter mirrors the qwen-image / flux-2 single-TE image
LoRA shape, with a few Ideogram-4-specific differences:

  - The DiT is distributed ONLY in quantized form (FP8/NVFP4); there are no
    BF16/FP16 DiT weights. ``--dit`` therefore always loads the pre-quantized
    FP8 checkpoint as the frozen base — there is NO ``--fp8_base`` flag (the FP8
    base is implicit). ``--dit_dtype`` (default bfloat16) is the compute dtype
    the FP8 weights are dequantized to and that the LoRA modules run in.
  - ``--vae`` is the Flux2 KL-VAE (``--vae_dtype bfloat16``) and ``--text_encoder``
    is Qwen3-VL-8B. Both are only strictly needed for sampling during training,
    but latent / TE pre-caching needs them, so we always wire all three weights.
  - Flow-matching uses ``--timestep_sampling ideogram4_shift`` (a resolution-aware
    logit-normal sampler aligned with the official pipeline) and ``--weighting_scheme
    none`` (anything else is rejected upstream). These are pinned, not searched.
  - LoRA REQUIRES ``--network_module networks.lora_ideogram4``; it trains only the
    conditional transformer (attention.qkv/o + feed_forward.w1/w2/w3).
  - ``--blocks_to_swap`` max is 33 (the DiT has 34 layers), so we clamp the
    tier-based swap table value to 33.

This Ideogram 4 support is EXPERIMENTAL upstream (see docs/ideogram4.md).

Module names used by musubi-tuner upstream (docs/ideogram4.md):
  train:      musubi_tuner.ideogram4_train_network
  cache lat:  musubi_tuner.ideogram4_cache_latents          (needs --vae)
  cache TE:   musubi_tuner.ideogram4_cache_text_encoder_outputs (needs --text_encoder)

batch_size lives in the dataset TOML (musubi convention), so we write a
per-candidate dataset TOML and point the trainer at it via ``--dataset_config``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.dataset.runtime import derive_run_toml
from bracket.hardware import (
    BLOCKS_TO_SWAP_BY_TIER,
    MUSUBI_DATALOADER_WORKERS_BY_TIER,
    SDXL_LORA_BATCH_CHOICES_BY_TIER,
    SDXL_LORA_DEFAULT_BATCH_BY_TIER,
    detect_gpu,
    lora_grad_ckpt_baseline,
    lora_grad_ckpt_varies,
    vram_tier,
)
from bracket.search.space import (
    CategoricalKnob,
    FixedKnob,
    FloatKnob,
    IntKnob,
    SearchSpace,
)
from bracket.trainer.base import (
    LaunchSpec,
    Trainer,
    TrainerConfig,
    make_accelerate_launch_prefix,
    make_subprocess_env,
    resolve_save_every_n_steps,
)

logger = logging.getLogger("bracket.trainer.ideogram4_lora")

_MODEL_CLASS = "ideogram4"

# The DiT is a 34-layer single-stream transformer; musubi caps --blocks_to_swap
# at 33 (docs/ideogram4.md → Memory Optimization). Clamp the tier table value.
_MAX_BLOCKS_TO_SWAP = 33


def _clamp_blocks_to_swap(tier: str) -> int:
    return min(BLOCKS_TO_SWAP_BY_TIER[tier], _MAX_BLOCKS_TO_SWAP)


@dataclass
class Ideogram4LoRAConfig(TrainerConfig):
    learning_rate: float = 1e-4
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 16.0
    train_batch_size: int = 1            # written into dataset TOML, not CLI
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    dit_dtype: str = "bfloat16"          # compute dtype; FP8 base is implicit
    vae_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    blocks_to_swap: int = 0
    dataloader_workers: int = 2


class Ideogram4LoRATrainer(Trainer):
    name = "ideogram4-lora"

    def __init__(
        self,
        *,
        musubi_dir: Path,
        venv_python: Path,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        vram_gb: Optional[float] = None,
    ) -> None:
        self.musubi_dir = Path(musubi_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        self.train_script = (
            self.musubi_dir / "src" / "musubi_tuner" / "ideogram4_train_network.py"
        )
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "ideogram4_train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"ideogram4_train_network.py not found under {self.musubi_dir}"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        if lora_grad_ckpt_varies(self.tier, _MODEL_CLASS):
            ckpt_knob: Any = CategoricalKnob(choices=(True, False))
        else:
            ckpt_knob = FixedKnob(value=True)
        return SearchSpace(
            name=f"ideogram4-lora-v0.1-{self.tier}",
            knobs={
                # loss-bearing knobs (the search actually moves these)
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
                "train_batch_size": CategoricalKnob(
                    choices=SDXL_LORA_BATCH_CHOICES_BY_TIER[self.tier],
                ),
                "gradient_checkpointing": ckpt_knob,
                # pinned — not loss-bearing
                "dataloader_workers": FixedKnob(
                    value=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]
                ),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
                "dit_dtype": FixedKnob(value="bfloat16"),
                "vae_dtype": FixedKnob(value="bfloat16"),
                "blocks_to_swap": FixedKnob(value=_clamp_blocks_to_swap(self.tier)),
            },
        )

    def baseline_config(self) -> Ideogram4LoRAConfig:
        return Ideogram4LoRAConfig(
            learning_rate=1e-4,
            optimizer_type="AdamW8bit",
            lr_scheduler="cosine",
            lr_warmup_steps=100,
            network_dim=32,
            network_alpha=16.0,
            train_batch_size=SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, _MODEL_CLASS),
            dataloader_workers=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier],
            blocks_to_swap=_clamp_blocks_to_swap(self.tier),
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, _MODEL_CLASS)
        workers = MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]
        swap = _clamp_blocks_to_swap(self.tier)
        # The orchestrator clamps LR at runtime, so curated LRs may sit outside
        # the search-space FloatKnob range (Prodigy's lr=1.0). Every other knob
        # stays inside the declared space so curated configs validate.
        return [
            # 1) Conservative starter — Ideogram 4 is a large DiT; 5e-5 AdamW8bit
            #    is a safe LoRA starter for big single-stream transformers.
            Ideogram4LoRAConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 2) Prodigy auto-LR — often wins on large DiT LoRA fine-tunes.
            Ideogram4LoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=16.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 3) Higher-rank adapter at the documented LR — more capacity for
            #    detailed subjects; Lion pairs well with large transformers.
            Ideogram4LoRAConfig(
                learning_rate=1e-4, optimizer_type="Lion",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=64, network_alpha=32.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> Ideogram4LoRAConfig:
        return Ideogram4LoRAConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer_type=str(knobs["optimizer_type"]),
            lr_scheduler=str(knobs["lr_scheduler"]),
            lr_warmup_steps=int(knobs["lr_warmup_steps"]),
            network_dim=int(knobs["network_dim"]),
            network_alpha=float(knobs["network_alpha"]),
            train_batch_size=int(knobs["train_batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            mixed_precision=str(knobs["mixed_precision"]),
            max_grad_norm=float(knobs["max_grad_norm"]),
            dit_dtype=str(knobs["dit_dtype"]),
            vae_dtype=str(knobs["vae_dtype"]),
            gradient_checkpointing=bool(knobs["gradient_checkpointing"]),
            blocks_to_swap=int(knobs["blocks_to_swap"]),
            dataloader_workers=int(knobs["dataloader_workers"]),
        )

    def session_setup_commands(
        self, *, dataset_toml: Path, run_dir: Path,
    ) -> list[LaunchSpec]:
        from bracket.trainer.flux2_klein_lora import _musubi_pre_cache_commands

        return _musubi_pre_cache_commands(
            musubi_dir=self.musubi_dir,
            venv_python=self.venv_python,
            run_dir=run_dir,
            dataset_toml=dataset_toml,
            cache_latents_module="musubi_tuner.ideogram4_cache_latents",
            cache_te_module="musubi_tuner.ideogram4_cache_text_encoder_outputs",
            vae_path=self.vae_path,
            text_encoder_path=self.text_encoder_path,
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
        if not isinstance(config, Ideogram4LoRAConfig):
            raise TypeError(
                f"expected Ideogram4LoRAConfig, got {type(config).__name__}"
            )
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "sample"  # musubi default — singular
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        # batch_size lives in the dataset TOML; musubi has no --train_batch_size.
        run_toml = derive_run_toml(
            source_toml=dataset_toml,
            target_path=run_dir / "dataset.toml",
            batch_size=config.train_batch_size,
            target_format="musubi",
        )

        cmd: list[str] = [
            *make_accelerate_launch_prefix(
                self.venv_python, mixed_precision=config.mixed_precision
            ),
            str(self.train_script),
            # Ideogram 4 single-TE image weights. --dit loads the FP8 frozen base
            # (no --fp8_base flag — FP8 is implicit); --dit_dtype is the compute
            # dtype. --vae / --text_encoder are needed for caching + sampling.
            "--dit", self.dit_path,
            "--dit_dtype", config.dit_dtype,
            "--vae", self.vae_path,
            "--vae_dtype", config.vae_dtype,
            "--text_encoder", self.text_encoder_path,
            "--dataset_config", str(run_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--network_module", "networks.lora_ideogram4",
            "--network_dim", str(config.network_dim),
            "--network_alpha", str(config.network_alpha),
            "--learning_rate", f"{config.learning_rate:.10g}",
            "--optimizer_type", config.optimizer_type,
            "--lr_scheduler", config.lr_scheduler,
            "--lr_warmup_steps", str(config.lr_warmup_steps),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--mixed_precision", config.mixed_precision,
            "--max_grad_norm", str(config.max_grad_norm),
            "--max_train_steps", str(max_steps),
            "--seed", str(seed),
            "--sdpa",
            # Ideogram 4 flow-matching: resolution-aware ideogram4_shift sampler,
            # plain MSE loss (weighting_scheme other than none is rejected).
            "--timestep_sampling", "ideogram4_shift",
            "--weighting_scheme", "none",
            "--max_data_loader_n_workers", str(config.dataloader_workers),
            "--persistent_data_loader_workers",
            "--save_every_n_steps",
            str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),
            "--no_metadata",
        ]
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
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
            cmd=cmd, cwd=self.musubi_dir, env=make_subprocess_env(),
            output_dir=output_dir, logging_dir=logging_dir,
            tfevents_glob=str(logging_dir / "**" / "events.out.tfevents.*"),
            sample_dir=sample_dir,
        )
