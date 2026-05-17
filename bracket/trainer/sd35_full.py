"""SD3.5 full fine-tune — wraps sd-scripts/sd3_train.py.

Same constraints as SDXL full FT (Adafactor + fused_backward_pass + full_bf16
+ --max_grad_norm 0). Larger model class than SDXL but flow-matching adds
--discrete_flow_shift + --weighting_scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.hardware import detect_gpu, vram_tier
from bracket.search.space import (
    CategoricalKnob, FixedKnob, FloatKnob, IntKnob, SearchSpace,
)
from bracket.trainer.base import (
    LaunchSpec, Trainer, TrainerConfig,
    make_accelerate_launch_prefix, make_subprocess_env, resolve_save_every_n_steps,
)


_BATCH_BY_TIER = {
    "xl": (1, 2, 4), "large": (1, 2), "high": (1, 2),
    "med": (1,), "low": (1,), "tiny": (1,),
}
_DEFAULT_BATCH_BY_TIER = {"xl": 4, "large": 2, "high": 2, "med": 1, "low": 1, "tiny": 1}


@dataclass
class SD35FullConfig(TrainerConfig):
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
    noise_offset: float = 0.0


class SD35FullTrainer(Trainer):
    name = "sd35-full-sd-scripts"

    def __init__(
        self,
        *,
        sd_scripts_dir: Path,
        venv_python: Path,
        pretrained_model: str,
        vram_gb: Optional[float] = None,
    ) -> None:
        self.sd_scripts_dir = Path(sd_scripts_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.pretrained_model = pretrained_model
        if not (self.sd_scripts_dir / "sd3_train.py").exists():
            raise FileNotFoundError(
                f"sd3_train.py not found in {self.sd_scripts_dir} (needs sd-scripts sd3 branch)"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(
            name=f"sd35-full-v0.1-{self.tier}",
            knobs={
                "learning_rate": FloatKnob(low=1e-7, high=5e-6, log=True),
                "optimizer_type": CategoricalKnob(choices=("Adafactor", "AdamW")),
                "lr_scheduler": CategoricalKnob(
                    choices=("constant_with_warmup", "constant", "cosine"),
                ),
                "lr_warmup_steps": IntKnob(low=10, high=200),
                "discrete_flow_shift": FloatKnob(low=2.0, high=4.0),
                "train_batch_size": CategoricalKnob(choices=_BATCH_BY_TIER[self.tier]),
                "noise_offset": FloatKnob(low=0.0, high=0.1),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "full_bf16": FixedKnob(value=True),
                "fused_backward_pass": FixedKnob(value=True),
                "max_grad_norm": FixedKnob(value=0.0),
            },
        )

    def baseline_config(self) -> SD35FullConfig:
        return SD35FullConfig(train_batch_size=_DEFAULT_BATCH_BY_TIER[self.tier])

    def curated_configs(self) -> list[TrainerConfig]:
        bs = _DEFAULT_BATCH_BY_TIER[self.tier]
        return [
            SD35FullConfig(
                learning_rate=1e-6, lr_scheduler="cosine",
                lr_warmup_steps=100, discrete_flow_shift=3.0,
                train_batch_size=bs,
            ),
            SD35FullConfig(
                learning_rate=2e-6, lr_scheduler="constant_with_warmup",
                lr_warmup_steps=50, discrete_flow_shift=3.0,
                train_batch_size=bs,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> SD35FullConfig:
        return SD35FullConfig(
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
            noise_offset=float(knobs["noise_offset"]),
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
        if not isinstance(config, SD35FullConfig):
            raise TypeError(f"expected SD35FullConfig, got {type(config).__name__}")
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "sample"
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            *make_accelerate_launch_prefix(self.venv_python, mixed_precision=config.mixed_precision),
            "sd3_train.py",
            "--pretrained_model_name_or_path", self.pretrained_model,
            "--dataset_config", str(dataset_toml),
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
            "--gradient_checkpointing",
            "--cache_latents",
            "--cache_latents_to_disk",
            "--sdpa",
            "--save_precision", "bf16",
            "--save_model_as", "safetensors",
            "--save_every_n_steps", str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
            "--weighting_scheme", "logit_normal",
            "--discrete_flow_shift", f"{config.discrete_flow_shift:.4f}",
        ]
        # Skip the per-image latent integrity check when the dataset is
        # already cached (saves several minutes on large sets).
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
