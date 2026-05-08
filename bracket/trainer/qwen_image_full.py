"""Qwen-Image full fine-tune trainer — wraps musubi-tuner qwen_image_train.

Same constraints as Z-Image full FT: Adafactor + fused_backward_pass +
full_bf16, --max_grad_norm 0, --mem_eff_save. The model is much larger (20B
vs 9B) so blocks_to_swap is essential below 80 GB.

Module names (musubi-tuner upstream):
  train:      musubi_tuner.qwen_image_train
  cache lat:  musubi_tuner.qwen_image_cache_latents
  cache TE:   musubi_tuner.qwen_image_cache_text_encoder_outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.dataset.runtime import derive_run_toml
from bracket.hardware import (
    BLOCKS_TO_SWAP_BY_TIER, MUSUBI_DATALOADER_WORKERS_BY_TIER,
    detect_gpu, vram_tier,
)
from bracket.search.space import (
    CategoricalKnob, FixedKnob, FloatKnob, IntKnob, SearchSpace,
)
from bracket.trainer.base import (
    LaunchSpec, Trainer, TrainerConfig,
    make_accelerate_launch_prefix, make_subprocess_env,
)


_BATCH_BY_TIER: dict[str, tuple[int, ...]] = {
    "xl":    (1, 2),
    "large": (1,),
    "high":  (1,),
    "med":   (1,),
    "low":   (1,),
    "tiny":  (1,),
}
_DEFAULT_BATCH_BY_TIER = {"xl": 2, "large": 1, "high": 1, "med": 1, "low": 1, "tiny": 1}
# Full-FT swap one tier higher than LoRA — optimizer state dominates.
_FULL_SWAP_BY_TIER = {
    "xl": 0, "large": 16, "high": 28, "med": 36, "low": 40, "tiny": 40,
}


@dataclass
class QwenImageFullConfig(TrainerConfig):
    learning_rate: float = 1e-6
    optimizer_type: str = "Adafactor"
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 50
    discrete_flow_shift: float = 3.0
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    full_bf16: bool = True
    fused_backward_pass: bool = True
    max_grad_norm: float = 0.0
    blocks_to_swap: int = 28


class QwenImageFullTrainer(Trainer):
    name = "qwen-image-full-musubi"

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
        self.train_script = self.musubi_dir / "src" / "musubi_tuner" / "qwen_image_train.py"
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "qwen_image_train.py"
        if not self.train_script.exists():
            raise FileNotFoundError(f"qwen_image_train.py not found under {self.musubi_dir}")
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(
            name=f"qwen-image-full-v0.1-{self.tier}",
            knobs={
                "learning_rate": FloatKnob(low=1e-7, high=1e-5, log=True),
                "optimizer_type": CategoricalKnob(choices=("Adafactor",)),
                "lr_scheduler": CategoricalKnob(
                    choices=("constant_with_warmup", "constant", "cosine"),
                ),
                "lr_warmup_steps": IntKnob(low=10, high=200),
                "discrete_flow_shift": FloatKnob(low=2.0, high=5.0),
                "train_batch_size": CategoricalKnob(choices=_BATCH_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "full_bf16": FixedKnob(value=True),
                "fused_backward_pass": FixedKnob(value=True),
                "max_grad_norm": FixedKnob(value=0.0),
                "blocks_to_swap": FixedKnob(value=_FULL_SWAP_BY_TIER[self.tier]),
            },
        )

    def baseline_config(self) -> QwenImageFullConfig:
        return QwenImageFullConfig(
            train_batch_size=_DEFAULT_BATCH_BY_TIER[self.tier],
            blocks_to_swap=_FULL_SWAP_BY_TIER[self.tier],
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = _DEFAULT_BATCH_BY_TIER[self.tier]
        swap = _FULL_SWAP_BY_TIER[self.tier]
        return [
            QwenImageFullConfig(
                learning_rate=5e-6, lr_scheduler="constant_with_warmup",
                lr_warmup_steps=50, discrete_flow_shift=3.0,
                train_batch_size=bs, blocks_to_swap=swap,
            ),
            QwenImageFullConfig(
                learning_rate=1e-5, lr_scheduler="cosine",
                lr_warmup_steps=200, discrete_flow_shift=3.0,
                train_batch_size=bs, blocks_to_swap=swap,
            ),
            QwenImageFullConfig(
                learning_rate=2e-6, lr_scheduler="cosine",
                lr_warmup_steps=100, discrete_flow_shift=4.0,
                train_batch_size=bs, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> QwenImageFullConfig:
        return QwenImageFullConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer_type=str(knobs["optimizer_type"]),
            lr_scheduler=str(knobs["lr_scheduler"]),
            lr_warmup_steps=int(knobs["lr_warmup_steps"]),
            discrete_flow_shift=float(knobs["discrete_flow_shift"]),
            train_batch_size=int(knobs["train_batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            mixed_precision=str(knobs["mixed_precision"]),
            full_bf16=bool(knobs["full_bf16"]),
            fused_backward_pass=bool(knobs["fused_backward_pass"]),
            max_grad_norm=float(knobs["max_grad_norm"]),
            blocks_to_swap=int(knobs["blocks_to_swap"]),
        )

    def session_setup_commands(self, *, dataset_toml: Path, run_dir: Path) -> list[LaunchSpec]:
        from bracket.trainer.flux2_klein_lora import _musubi_pre_cache_commands
        return _musubi_pre_cache_commands(
            musubi_dir=self.musubi_dir,
            venv_python=self.venv_python,
            run_dir=run_dir,
            dataset_toml=dataset_toml,
            cache_latents_module="musubi_tuner.qwen_image_cache_latents",
            cache_te_module="musubi_tuner.qwen_image_cache_text_encoder_outputs",
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
    ) -> LaunchSpec:
        if not isinstance(config, QwenImageFullConfig):
            raise TypeError(f"expected QwenImageFullConfig, got {type(config).__name__}")
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "sample"
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

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
            "--learning_rate", f"{config.learning_rate:.10g}",
            "--optimizer_type", config.optimizer_type,
            "--lr_scheduler", config.lr_scheduler,
            "--lr_warmup_steps", str(config.lr_warmup_steps),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--mixed_precision", config.mixed_precision,
            "--max_grad_norm", str(config.max_grad_norm),
            "--max_train_steps", str(max_steps),
            "--seed", str(seed),
            "--gradient_checkpointing",
            "--sdpa",
            "--timestep_sampling", "shift",
            "--weighting_scheme", "none",
            "--discrete_flow_shift", f"{config.discrete_flow_shift:.4f}",
            "--max_data_loader_n_workers", str(MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]),
            "--persistent_data_loader_workers",
            "--save_every_n_steps", str(max(1, max_steps)),
            "--no_metadata",
            "--mem_eff_save",
        ]
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
        return LaunchSpec(
            cmd=cmd, cwd=self.musubi_dir, env=make_subprocess_env(),
            output_dir=output_dir, logging_dir=logging_dir,
            tfevents_glob=str(logging_dir / "**" / "events.out.tfevents.*"),
            sample_dir=sample_dir,
        )
