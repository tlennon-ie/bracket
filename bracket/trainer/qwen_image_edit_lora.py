"""Qwen-Image-Edit LoRA — wraps musubi-tuner qwen_image_train_network.

Edit variant takes a paired (source_image, target_image) dataset shape rather
than (image, caption). Upstream musubi-tuner unified the edit path onto the
SAME scripts as plain Qwen-Image and selects the architecture with the
``--model_version`` flag (``edit`` / ``edit-2509`` / ``edit-2511``) instead of
shipping separate ``qwen_image_edit_*`` scripts. Without ``--model_version`` the
unified script defaults to ``original`` (plain Qwen-Image), which would train
the wrong architecture against Edit weights — so the flag is mandatory here.

  - ``edit``       — original Qwen-Image-Edit (single control image).
  - ``edit-2509``  — Edit-2509: multi-reference editing (up to 3 control images).
  - ``edit-2511``  — Edit-2511: multi-reference editing, newer checkpoint.

Module names (musubi-tuner upstream — shared with plain Qwen-Image):
  train:      musubi_tuner.qwen_image_train_network        (+ --model_version)
  cache lat:  musubi_tuner.qwen_image_cache_latents        (+ --model_version)
  cache TE:   musubi_tuner.qwen_image_cache_text_encoder_outputs (+ --model_version)

The dataset TOML for edit must declare both `image_directory` (target) and
`control_directory` (source). Bracket leaves that as the user's responsibility
— we don't synthesize edit pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.dataset.runtime import derive_run_toml
from bracket.hardware import (
    BLOCKS_TO_SWAP_BY_TIER, MUSUBI_DATALOADER_WORKERS_BY_TIER,
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
from bracket.trainer.qwen_image_lora import QwenImageLoRAConfig


_VALID_MODEL_VERSIONS = ("edit", "edit-2509", "edit-2511")


@dataclass
class QwenImageEditLoRAConfig(QwenImageLoRAConfig):
    """Identical knob shape to QwenImageLoRAConfig — edit conditioning lives in
    the dataset TOML, not the CLI."""


class QwenImageEditLoRATrainer(Trainer):
    name = "qwen-image-edit-lora-musubi"

    def __init__(
        self,
        *,
        musubi_dir: Path,
        venv_python: Path,
        dit_path: str,
        vae_path: str,
        text_encoder_path: str,
        model_version: str = "edit-2509",
        vram_gb: Optional[float] = None,
    ) -> None:
        if model_version not in _VALID_MODEL_VERSIONS:
            raise ValueError(
                f"model_version must be one of {_VALID_MODEL_VERSIONS}, got {model_version!r}"
            )
        self.model_version = model_version
        self.musubi_dir = Path(musubi_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_path = text_encoder_path
        # Upstream unified the edit path onto the plain Qwen-Image scripts;
        # the Edit architecture is selected with --model_version, not a
        # separate qwen_image_edit_train_network.py (which no longer exists).
        self.train_script = self.musubi_dir / "src" / "musubi_tuner" / "qwen_image_train_network.py"
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "qwen_image_train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"qwen_image_train_network.py not found under {self.musubi_dir}"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        if lora_grad_ckpt_varies(self.tier, "qwen-image-edit"):
            ckpt_knob = CategoricalKnob(choices=(True, False))
        else:
            ckpt_knob = FixedKnob(value=True)
        return SearchSpace(
            name=f"qwen-image-edit-lora-v0.1-{self.tier}",
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
                "discrete_flow_shift": FloatKnob(low=2.0, high=5.0),
                "train_batch_size": CategoricalKnob(
                    choices=SDXL_LORA_BATCH_CHOICES_BY_TIER[self.tier],
                ),
                "gradient_checkpointing": ckpt_knob,
                "dataloader_workers": FixedKnob(value=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
                "fp8_base": FixedKnob(value=True),
                "fp8_scaled": FixedKnob(value=True),
                "fp8_vl": FixedKnob(value=self.tier in ("low", "tiny", "med")),
                "blocks_to_swap": FixedKnob(value=BLOCKS_TO_SWAP_BY_TIER[self.tier]),
            },
        )

    def baseline_config(self) -> QwenImageEditLoRAConfig:
        return QwenImageEditLoRAConfig(
            learning_rate=1e-4,
            train_batch_size=SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, "qwen-image-edit"),
            dataloader_workers=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier],
            blocks_to_swap=BLOCKS_TO_SWAP_BY_TIER[self.tier],
            fp8_vl=self.tier in ("low", "tiny", "med"),
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, "qwen-image-edit")
        workers = MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]
        swap = BLOCKS_TO_SWAP_BY_TIER[self.tier]
        return [
            QwenImageEditLoRAConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            QwenImageEditLoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> QwenImageEditLoRAConfig:
        return QwenImageEditLoRAConfig(
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
            fp8_base=bool(knobs["fp8_base"]),
            fp8_scaled=bool(knobs["fp8_scaled"]),
            fp8_vl=bool(knobs["fp8_vl"]),
            gradient_checkpointing=bool(knobs["gradient_checkpointing"]),
            blocks_to_swap=int(knobs["blocks_to_swap"]),
            dataloader_workers=int(knobs["dataloader_workers"]),
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
            extra_args=["--model_version", self.model_version],
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
        if not isinstance(config, QwenImageEditLoRAConfig):
            raise TypeError(f"expected QwenImageEditLoRAConfig, got {type(config).__name__}")
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
            "--model_version", self.model_version,
            "--dataset_config", str(run_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--network_module", "networks.lora_qwen_image",
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
            "--save_every_n_steps", str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),
            "--no_metadata",
        ]
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if config.fp8_base:
            cmd.append("--fp8_base")
        if config.fp8_scaled:
            cmd.append("--fp8_scaled")
        if config.fp8_vl:
            cmd.append("--fp8_vl")
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
