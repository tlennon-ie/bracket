"""SD3.5 LoRA — wraps sd-scripts/sd3_train_network.py.

SD3.5 (Medium ~2.5B / Large ~8B) is a Stability MMDiT. sd-scripts handles it
via the sd3 branch with `sd3_train_network.py` (LoRA) and `sd3_train.py`
(full FT). The training surface mirrors SDXL with flow-matching extras.

Pass `pretrained_model` as a Stability HF folder OR a single `.safetensors`
that bundles MMDiT + T5-XXL + CLIP-L + CLIP-G — sd-scripts auto-loads from
the bundle. Override TE paths only if you want to swap encoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.hardware import (
    DATALOADER_WORKERS_BY_TIER,
    SDXL_LORA_BATCH_CHOICES_BY_TIER,
    SDXL_LORA_DEFAULT_BATCH_BY_TIER,
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
class SD35LoRAConfig(TrainerConfig):
    learning_rate: float = 1e-4
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 16.0
    discrete_flow_shift: float = 3.0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    noise_offset: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    dataloader_workers: int = 2


class SD35LoRATrainer(Trainer):
    name = "sd35-lora-sd-scripts"

    def __init__(
        self,
        *,
        sd_scripts_dir: Path,
        venv_python: Path,
        pretrained_model: str,
        model_class: str = "sd35-medium",
        vram_gb: Optional[float] = None,
    ) -> None:
        self.sd_scripts_dir = Path(sd_scripts_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.pretrained_model = pretrained_model
        # `sd35-medium` (~2.5B) or `sd35-large` (~8B) — drives ckpt-baseline
        # logic via MODEL_PARAMS_B.
        self.model_class = model_class
        if not (self.sd_scripts_dir / "sd3_train_network.py").exists():
            raise FileNotFoundError(
                f"sd3_train_network.py not found in {self.sd_scripts_dir} "
                "(needs sd-scripts sd3 branch)"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        if lora_grad_ckpt_varies(self.tier, self.model_class):
            ckpt_knob = CategoricalKnob(choices=(True, False))
        else:
            ckpt_knob = FixedKnob(value=True)
        return SearchSpace(
            name=f"sd35-lora-v0.1-{self.tier}",
            knobs={
                "learning_rate": FloatKnob(low=1e-6, high=5e-4, log=True),
                "optimizer_type": CategoricalKnob(
                    choices=("AdamW8bit", "AdamW", "Lion", "Prodigy", "Adafactor"),
                ),
                "lr_scheduler": CategoricalKnob(
                    choices=("cosine", "constant", "constant_with_warmup", "linear"),
                ),
                "lr_warmup_steps": IntKnob(low=0, high=200),
                "network_dim": CategoricalKnob(choices=(8, 16, 32, 64)),
                "network_alpha": CategoricalKnob(choices=(4.0, 8.0, 16.0, 32.0)),
                "discrete_flow_shift": FloatKnob(low=2.0, high=4.0),
                "noise_offset": FloatKnob(low=0.0, high=0.1),
                "train_batch_size": CategoricalKnob(
                    choices=SDXL_LORA_BATCH_CHOICES_BY_TIER[self.tier],
                ),
                "gradient_checkpointing": ckpt_knob,
                "dataloader_workers": FixedKnob(value=DATALOADER_WORKERS_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
            },
        )

    def baseline_config(self) -> SD35LoRAConfig:
        return SD35LoRAConfig(
            train_batch_size=SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, self.model_class),
            dataloader_workers=DATALOADER_WORKERS_BY_TIER[self.tier],
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, self.model_class)
        workers = DATALOADER_WORKERS_BY_TIER[self.tier]
        return [
            SD35LoRAConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
            SD35LoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> SD35LoRAConfig:
        return SD35LoRAConfig(
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
        if not isinstance(config, SD35LoRAConfig):
            raise TypeError(f"expected SD35LoRAConfig, got {type(config).__name__}")
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "sample"
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            *make_accelerate_launch_prefix(self.venv_python, mixed_precision=config.mixed_precision),
            "sd3_train_network.py",
            "--pretrained_model_name_or_path", self.pretrained_model,
            "--dataset_config", str(dataset_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--network_module", "networks.lora_sd3",
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
            "--sdpa",
            "--save_precision", "bf16",
            "--save_model_as", "safetensors",
            "--no_metadata",
            "--save_every_n_steps", str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),
            "--max_data_loader_n_workers", str(config.dataloader_workers),
            "--persistent_data_loader_workers",
            "--weighting_scheme", "logit_normal",
            "--discrete_flow_shift", f"{config.discrete_flow_shift:.4f}",
        ]
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if config.noise_offset and config.noise_offset > 0:
            cmd += ["--noise_offset", f"{config.noise_offset:.6f}"]
        if sample_prompts is not None and sample_every_n_steps:
            cmd += [
                "--sample_prompts", str(sample_prompts),
                "--sample_every_n_steps", str(sample_every_n_steps),
                "--sample_sampler", "euler",
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
