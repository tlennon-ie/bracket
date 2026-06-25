"""Krea 2 LoRA trainer adapter — wraps musubi-tuner krea2_train_network.

Krea 2 (K2) is a single-stream MMDiT text-to-image model. It is, for Bracket's
purposes, a near-exact sibling of the Qwen-Image integration:
  - batch_size lives in the dataset TOML (not on the CLI)
  - latents and TE outputs must be pre-cached via separate scripts
  - flow-matching with --timestep_sampling + --discrete_flow_shift
  - fp8_base / fp8_scaled both available (K2 rejects plain fp8 without scaled);
    plus blocks_to_swap if VRAM is tight (max 26 = 28 main blocks − 2).

Weights (see vendor/musubi-tuner/docs/krea2.md):
  - --dit: the Krea 2 **RAW** model (train on RAW, infer on Turbo).
  - --vae: the **Qwen-Image VAE** — the SAME file the Qwen-Image integration
    uses; latent normalization is identical.
  - --text_encoder: **Qwen3-VL-4B-Instruct** (single TE, single-file
    safetensors). Only required by the TE pre-cache step and (optionally) for
    sample-image generation; the train step itself reads pre-cached TE outputs.

Module names used by musubi-tuner upstream (pinned v0.3.4):
  train:      musubi_tuner.krea2_train_network  (networks.lora_krea2)
  cache lat:  musubi_tuner.krea2_cache_latents              (needs --vae)
  cache TE:   musubi_tuner.krea2_cache_text_encoder_outputs (needs --text_encoder)

Timestep sampling: we use the documented starting point
``--timestep_sampling shift`` with a searchable ``--discrete_flow_shift``
centered on 2.5 (2.5 matches K2's 1024px inference time-shift). The
resolution-aware ``krea2_shift`` alternative is available upstream but pins the
schedule and removes the searchable knob, so we expose ``shift`` instead.
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
    LaunchSpec, Trainer, TrainerConfig,
    make_accelerate_launch_prefix, make_subprocess_env, resolve_save_every_n_steps,
)

# Model class key for VRAM-headroom heuristics. Krea 2's 28 main blocks land it
# in the big-DiT regime; ``MODEL_PARAMS_B.get(..., 10.0)`` already treats an
# unregistered key as a large model, so no hardware.py change is required.
_MODEL_CLASS = "krea2"

logger = logging.getLogger("bracket.trainer.krea2_lora")

# Krea 2 has 28 main SingleStreamBlocks; musubi caps ``--blocks_to_swap`` at 26
# (28 − 2). Clamp the tier-derived value so low-VRAM tiers don't emit a value
# musubi rejects at startup.
_MAX_BLOCKS_TO_SWAP = 26


def _clamp_blocks_to_swap(tier: str) -> int:
    return min(BLOCKS_TO_SWAP_BY_TIER[tier], _MAX_BLOCKS_TO_SWAP)


@dataclass
class Krea2LoRAConfig(TrainerConfig):
    learning_rate: float = 1e-4
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 32.0          # K2 authors' recommended default
    discrete_flow_shift: float = 2.5     # matches K2 1024px inference time-shift
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    fp8_base: bool = True
    fp8_scaled: bool = True
    gradient_checkpointing: bool = True
    blocks_to_swap: int = 0
    dataloader_workers: int = 2


class Krea2LoRATrainer(Trainer):
    name = "krea2-lora"

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
        self.train_script = self.musubi_dir / "src" / "musubi_tuner" / "krea2_train_network.py"
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "krea2_train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"krea2_train_network.py not found under {self.musubi_dir}"
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
            name=f"krea2-lora-v0.1-{self.tier}",
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
                # Centered on 2.5 (K2 1024px shift); spans the resolution-aware
                # range the doc cites (~1.6 at 256px to ~3.2 at 1280px).
                "discrete_flow_shift": FloatKnob(low=1.6, high=3.2),
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
                "blocks_to_swap": FixedKnob(value=_clamp_blocks_to_swap(self.tier)),
            },
        )

    def baseline_config(self) -> Krea2LoRAConfig:
        return Krea2LoRAConfig(
            learning_rate=1e-4,
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
        return [
            # 1) Authors' recommended default — rank/alpha 32, shift 2.5.
            Krea2LoRAConfig(
                learning_rate=1e-4, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=32.0,
                discrete_flow_shift=2.5,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 2) Prodigy auto-LR — usually a strong contender on big DiTs.
            Krea2LoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=32.0,
                discrete_flow_shift=2.5,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 3) Conservative starter — lower LR with warmup, 16/16 LoRA.
            Krea2LoRAConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=16, network_alpha=16.0,
                discrete_flow_shift=2.5,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> Krea2LoRAConfig:
        return Krea2LoRAConfig(
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
            cache_latents_module="musubi_tuner.krea2_cache_latents",
            cache_te_module="musubi_tuner.krea2_cache_text_encoder_outputs",
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
        if not isinstance(config, Krea2LoRAConfig):
            raise TypeError(f"expected Krea2LoRAConfig, got {type(config).__name__}")
        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "sample"  # musubi default — singular
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
            "--network_module", "networks.lora_krea2",
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
