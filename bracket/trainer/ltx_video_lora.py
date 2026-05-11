"""LTX-Video LoRA — wraps musubi-tuner ltxv_train_network.

Lightricks LTX-Video is a small (~2B) video DiT — much cheaper to fine-tune
than HunyuanVideo / Wan. Single text encoder (T5-XXL).

Module names (musubi-tuner upstream):
  train:      musubi_tuner.ltxv_train_network
  cache lat:  musubi_tuner.ltxv_cache_latents
  cache TE:   musubi_tuner.ltxv_cache_text_encoder_outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bracket.dataset.runtime import derive_run_toml
from bracket.hardware import (
    BLOCKS_TO_SWAP_BY_TIER, MUSUBI_DATALOADER_WORKERS_BY_TIER,
    VIDEO_LORA_BATCH_CHOICES_BY_TIER, VIDEO_LORA_DEFAULT_BATCH_BY_TIER,
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
class LTXVideoLoRAConfig(TrainerConfig):
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
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    blocks_to_swap: int = 0
    dataloader_workers: int = 2


class LTXVideoLoRATrainer(Trainer):
    name = "ltx-video-lora-musubi"

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
        self.train_script = self.musubi_dir / "src" / "musubi_tuner" / "ltxv_train_network.py"
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "ltxv_train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"ltxv_train_network.py not found under {self.musubi_dir}"
            )
        if not self.venv_python.exists():
            raise FileNotFoundError(f"venv python not found: {self.venv_python}")
        if vram_gb is None:
            info = detect_gpu()
            vram_gb = info.total_vram_gb if info else 12.0
        self.vram_gb = float(vram_gb)
        self.tier = vram_tier(self.vram_gb)

    def declare_search_space(self) -> SearchSpace:
        if lora_grad_ckpt_varies(self.tier, "ltx-video"):
            ckpt_knob = CategoricalKnob(choices=(True, False))
        else:
            ckpt_knob = FixedKnob(value=True)
        return SearchSpace(
            name=f"ltx-video-lora-v0.1-{self.tier}",
            knobs={
                "learning_rate": FloatKnob(low=1e-5, high=5e-4, log=True),
                "optimizer_type": CategoricalKnob(
                    choices=("AdamW8bit", "AdamW", "Lion", "Prodigy"),
                ),
                "lr_scheduler": CategoricalKnob(
                    choices=("cosine", "constant", "constant_with_warmup"),
                ),
                "lr_warmup_steps": IntKnob(low=0, high=200),
                "network_dim": CategoricalKnob(choices=(16, 32, 64)),
                "network_alpha": CategoricalKnob(choices=(8.0, 16.0, 32.0)),
                "discrete_flow_shift": FloatKnob(low=2.0, high=5.0),
                "train_batch_size": CategoricalKnob(
                    choices=VIDEO_LORA_BATCH_CHOICES_BY_TIER[self.tier],
                ),
                "gradient_checkpointing": ckpt_knob,
                "dataloader_workers": FixedKnob(value=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
                "blocks_to_swap": FixedKnob(value=BLOCKS_TO_SWAP_BY_TIER[self.tier] // 2),
            },
        )

    def baseline_config(self) -> LTXVideoLoRAConfig:
        return LTXVideoLoRAConfig(
            train_batch_size=VIDEO_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, "ltx-video"),
            dataloader_workers=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier],
            blocks_to_swap=BLOCKS_TO_SWAP_BY_TIER[self.tier] // 2,
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = VIDEO_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, "ltx-video")
        workers = MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]
        swap = BLOCKS_TO_SWAP_BY_TIER[self.tier] // 2
        return [
            LTXVideoLoRAConfig(
                learning_rate=1e-4, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            LTXVideoLoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=3.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> LTXVideoLoRAConfig:
        return LTXVideoLoRAConfig(
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
            blocks_to_swap=int(knobs["blocks_to_swap"]),
            dataloader_workers=int(knobs["dataloader_workers"]),
        )

    def session_setup_commands(self, *, dataset_toml: Path, run_dir: Path) -> list[LaunchSpec]:
        env = make_subprocess_env()
        run_dir = Path(run_dir).resolve()
        logging_dir = run_dir / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)
        flat_toml = derive_run_toml(
            source_toml=dataset_toml,
            target_path=run_dir / "dataset_flat.toml",
            target_format="musubi",
        )
        return [
            LaunchSpec(
                cmd=[
                    str(self.venv_python), "-m", "musubi_tuner.ltxv_cache_latents",
                    "--dataset_config", str(flat_toml),
                    "--vae", self.vae_path,
                ],
                cwd=self.musubi_dir, env=env,
                output_dir=run_dir, logging_dir=logging_dir,
                tfevents_glob="",
            ),
            LaunchSpec(
                cmd=[
                    str(self.venv_python), "-m", "musubi_tuner.ltxv_cache_text_encoder_outputs",
                    "--dataset_config", str(flat_toml),
                    "--text_encoder", self.text_encoder_path,
                ],
                cwd=self.musubi_dir, env=env,
                output_dir=run_dir, logging_dir=logging_dir,
                tfevents_glob="",
            ),
        ]

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
        if not isinstance(config, LTXVideoLoRAConfig):
            raise TypeError(f"expected LTXVideoLoRAConfig, got {type(config).__name__}")
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
            "--network_module", "networks.lora",
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
