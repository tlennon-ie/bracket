"""Z-Image LoRA trainer adapter — wraps musubi-tuner zimage_train_network.

Real musubi arg surface (verified against `python -m musubi_tuner.zimage_train_network --help`):
  - NO --train_batch_size, NO --noise_offset, NO --min_snr_gamma
  - NO on-the-fly --cache_latents / --cache_text_encoder_outputs;
    musubi uses separate pre-caching scripts (zimage_cache_latents.py /
    zimage_cache_text_encoder_outputs.py). We document this in run_help_hint().
  - batch_size lives in the dataset TOML at the [[datasets]] level
  - flow-matching specific: --timestep_sampling, --discrete_flow_shift,
    --weighting_scheme

To let the orchestrator search batch_size, we write a per-candidate dataset
TOML inside the run directory with batch_size set, then point the trainer
at it via --dataset_config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.dataset.runtime import derive_run_toml
from bracket.hardware import (
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
    LaunchSpec, Trainer, TrainerConfig,
    make_accelerate_launch_prefix, make_subprocess_env,
)


@dataclass
class ZImageLoRAConfig(TrainerConfig):
    learning_rate: float = 1e-4
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 16.0
    discrete_flow_shift: float = 2.0
    train_batch_size: int = 1     # written into dataset TOML, not CLI
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    dataloader_workers: int = 2


class ZImageLoRATrainer(Trainer):
    name = "zimage-lora-musubi"

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
        self.train_script = self.musubi_dir / "src" / "musubi_tuner" / "zimage_train_network.py"
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "zimage_train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"zimage_train_network.py not found under {self.musubi_dir}"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        if lora_grad_ckpt_varies(self.tier, "zimage"):
            ckpt_knob = CategoricalKnob(choices=(True, False))
        else:
            ckpt_knob = FixedKnob(value=True)
        return SearchSpace(
            name=f"zimage-lora-v0.4-{self.tier}",
            knobs={
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
                "discrete_flow_shift": FloatKnob(low=1.0, high=4.0),
                "train_batch_size": CategoricalKnob(
                    choices=SDXL_LORA_BATCH_CHOICES_BY_TIER[self.tier]
                ),
                "gradient_checkpointing": ckpt_knob,
                "dataloader_workers": FixedKnob(value=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
            },
        )

    def baseline_config(self) -> ZImageLoRAConfig:
        return ZImageLoRAConfig(
            learning_rate=1e-4,
            optimizer_type="AdamW8bit",
            lr_scheduler="cosine",
            lr_warmup_steps=100,
            network_dim=32,
            network_alpha=16.0,
            discrete_flow_shift=2.0,
            train_batch_size=SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, "zimage"),
            dataloader_workers=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier],
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, "zimage")
        workers = MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]
        return [
            # 1) Lower LR + warmup — safer starting point if 1e-4 diverges
            #    on a particular dataset.
            ZImageLoRAConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=2.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
            # 2) Prodigy auto-LR — popular community recipe for "just works".
            ZImageLoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=2.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
            # 3) Higher flow_shift for portrait/influencer datasets.
            ZImageLoRAConfig(
                learning_rate=1e-4, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> ZImageLoRAConfig:
        return ZImageLoRAConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer_type=str(knobs["optimizer_type"]),
            lr_scheduler=str(knobs["lr_scheduler"]),
            lr_warmup_steps=int(knobs["lr_warmup_steps"]),
            network_dim=int(knobs["network_dim"]),
            network_alpha=float(knobs["network_alpha"]),
            discrete_flow_shift=float(knobs["discrete_flow_shift"]),
            train_batch_size=int(knobs["train_batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            mixed_precision=str(knobs["mixed_precision"]),
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
    ) -> LaunchSpec:
        if not isinstance(config, ZImageLoRAConfig):
            raise TypeError(f"expected ZImageLoRAConfig, got {type(config).__name__}")
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "sample"  # musubi default — singular
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Write per-run TOML with batch_size injected; musubi has no
        # --train_batch_size CLI flag.
        run_toml = derive_run_toml(
            source_toml=dataset_toml,
            target_path=run_dir / "dataset.toml",
            batch_size=config.train_batch_size,
            target_format="musubi",
        )

        cmd: list[str] = [
            *make_accelerate_launch_prefix(self.venv_python, mixed_precision=config.mixed_precision),
            str(self.train_script),
            "--dit", self.dit_path,
            "--vae", self.vae_path,
            "--text_encoder", self.text_encoder_path,
            "--dataset_config", str(run_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--network_module", "networks.lora_zimage",
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
            "--timestep_sampling", "shift",
            "--weighting_scheme", "none",
            "--discrete_flow_shift", f"{config.discrete_flow_shift:.4f}",
            "--max_data_loader_n_workers", str(config.dataloader_workers),
            "--persistent_data_loader_workers",
            "--save_every_n_steps", str(max(1, max_steps)),
            "--no_metadata",
        ]
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if sample_prompts is not None and sample_every_n_steps:
            cmd += [
                "--sample_prompts", str(sample_prompts),
                "--sample_every_n_steps", str(sample_every_n_steps),
            ]

        return LaunchSpec(
            cmd=cmd, cwd=self.musubi_dir, env=make_subprocess_env(),
            output_dir=output_dir, logging_dir=logging_dir,
            tfevents_glob=str(logging_dir / "**" / "events.out.tfevents.*"),
            sample_dir=sample_dir,
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
            cache_latents_module="musubi_tuner.zimage_cache_latents",
            cache_te_module="musubi_tuner.zimage_cache_text_encoder_outputs",
            vae_path=self.vae_path,
            text_encoder_path=self.text_encoder_path,
        )

    @staticmethod
    def pre_caching_hint() -> str:
        """Z-Image needs latents and TE outputs cached BEFORE training. Run once
        per session (or per dataset change) before invoking the orchestrator:

            python -m musubi_tuner.zimage_cache_latents \\
                --dataset_config <subset/dataset.toml> --vae <vae>
            python -m musubi_tuner.zimage_cache_text_encoder_outputs \\
                --dataset_config <subset/dataset.toml> --text_encoder <te>

        Without pre-cached latents/TE outputs, every candidate run repeats
        the VAE + TE encode work — slow but functional.
        """
        return ZImageLoRATrainer.pre_caching_hint.__doc__ or ""
