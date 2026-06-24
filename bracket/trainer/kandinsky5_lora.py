"""Kandinsky 5 LoRA trainer adapter — wraps musubi-tuner kandinsky5_train_network.

Kandinsky 5 is Sber's flow-matching model. The musubi adapter follows the same
conventions as Z-Image / Qwen-Image / Flux-2:
  - batch_size lives in the dataset TOML (not on the CLI)
  - latents and TE outputs must be pre-cached via separate scripts
  - flow-matching with --discrete_flow_shift + --timestep_sampling

Kandinsky 5 is unusual in the musubi family because it uses **two** text
encoders, each with its own CLI flag (verified against upstream
docs/kandinsky5.md):
  - --text_encoder_qwen  → Qwen2.5-VL-7B-Instruct
  - --text_encoder_clip  → CLIP ViT-Large (openai/clip-vit-large-patch14)

Module names used by musubi-tuner upstream:
  train:      musubi_tuner.kandinsky5_train_network
  cache lat:  musubi_tuner.kandinsky5_cache_latents
  cache TE:   musubi_tuner.kandinsky5_cache_text_encoder_outputs

A --task selector picks the model config/architecture. The default targets the
text-to-image-style pretrain config; the orchestrator pins it (non-loss-bearing).
"""

from __future__ import annotations

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

# Model-class key for the hardware heuristics. Unknown to MODEL_PARAMS_B, which
# falls back to the "assume big" 10B default — appropriate for Kandinsky 5.
_MODEL_CLASS = "kandinsky5"
# Default --task config (Kandinsky 5 Pro text→image pretrain). Pinned by the
# orchestrator; overridable at construction time.
_DEFAULT_TASK = "k5-pro-t2v-5s-sd"


@dataclass
class Kandinsky5LoRAConfig(TrainerConfig):
    learning_rate: float = 1e-4
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 16.0
    discrete_flow_shift: float = 5.0
    train_batch_size: int = 1            # written into dataset TOML, not CLI
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    fp8_base: bool = True
    gradient_checkpointing: bool = True
    blocks_to_swap: int = 0
    dataloader_workers: int = 2


class Kandinsky5LoRATrainer(Trainer):
    name = "kandinsky5-lora"

    def __init__(
        self,
        *,
        musubi_dir: Path,
        venv_python: Path,
        dit_path: str,
        vae_path: str,
        text_encoder_qwen_path: str,
        text_encoder_clip_path: str,
        task: str = _DEFAULT_TASK,
        vram_gb: Optional[float] = None,
    ) -> None:
        self.musubi_dir = Path(musubi_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.dit_path = dit_path
        self.vae_path = vae_path
        self.text_encoder_qwen_path = text_encoder_qwen_path
        self.text_encoder_clip_path = text_encoder_clip_path
        self.task = task
        self.train_script = self.musubi_dir / "src" / "musubi_tuner" / "kandinsky5_train_network.py"
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "kandinsky5_train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"kandinsky5_train_network.py not found under {self.musubi_dir}"
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
            name=f"kandinsky5-lora-v0.1-{self.tier}",
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
                "discrete_flow_shift": FloatKnob(low=3.0, high=7.0),
                "train_batch_size": CategoricalKnob(
                    choices=SDXL_LORA_BATCH_CHOICES_BY_TIER[self.tier],
                ),
                "gradient_checkpointing": ckpt_knob,
                "dataloader_workers": FixedKnob(value=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "mixed_precision": FixedKnob(value="bf16"),
                "max_grad_norm": FixedKnob(value=1.0),
                "fp8_base": FixedKnob(value=True),
                "blocks_to_swap": FixedKnob(value=BLOCKS_TO_SWAP_BY_TIER[self.tier]),
            },
        )

    def baseline_config(self) -> Kandinsky5LoRAConfig:
        return Kandinsky5LoRAConfig(
            learning_rate=1e-4,
            optimizer_type="AdamW8bit",
            lr_scheduler="cosine",
            lr_warmup_steps=100,
            network_dim=32,
            network_alpha=16.0,
            discrete_flow_shift=5.0,
            train_batch_size=SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier],
            gradient_checkpointing=lora_grad_ckpt_baseline(self.tier, _MODEL_CLASS),
            dataloader_workers=MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier],
            blocks_to_swap=BLOCKS_TO_SWAP_BY_TIER[self.tier],
        )

    def curated_configs(self) -> list[TrainerConfig]:
        bs = SDXL_LORA_DEFAULT_BATCH_BY_TIER[self.tier]
        ckpt = lora_grad_ckpt_baseline(self.tier, _MODEL_CLASS)
        workers = MUSUBI_DATALOADER_WORKERS_BY_TIER[self.tier]
        swap = BLOCKS_TO_SWAP_BY_TIER[self.tier]
        return [
            # 1) Conservative starter — lower LR with warmup; safe if 1e-4
            #    diverges on a particular dataset.
            Kandinsky5LoRAConfig(
                learning_rate=5e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=5.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 2) Prodigy auto-LR — "just works" community recipe.
            Kandinsky5LoRAConfig(
                learning_rate=1.0, optimizer_type="Prodigy",
                lr_scheduler="cosine", lr_warmup_steps=0,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=5.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 3) Higher flow_shift — Kandinsky's default shift is high (5.0);
            #    push toward the top of the range for detail-heavy datasets.
            Kandinsky5LoRAConfig(
                learning_rate=1e-4, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                discrete_flow_shift=6.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> Kandinsky5LoRAConfig:
        return Kandinsky5LoRAConfig(
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
            gradient_checkpointing=bool(knobs["gradient_checkpointing"]),
            blocks_to_swap=int(knobs["blocks_to_swap"]),
            dataloader_workers=int(knobs["dataloader_workers"]),
        )

    def session_setup_commands(
        self, *, dataset_toml: Path, run_dir: Path,
    ) -> list[LaunchSpec]:
        """Pre-cache latents and (dual) text-encoder outputs once per session.

        Kandinsky 5 needs both Qwen2.5-VL and CLIP outputs cached, so we can't
        reuse the single-TE musubi pre-cache helper; we build the two commands
        here with both --text_encoder_qwen and --text_encoder_clip.
        """
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
                    str(self.venv_python), "-m", "musubi_tuner.kandinsky5_cache_latents",
                    "--dataset_config", str(flat_toml),
                    "--vae", self.vae_path,
                ],
                cwd=self.musubi_dir, env=env,
                output_dir=run_dir, logging_dir=logging_dir,
                tfevents_glob="",
            ),
            LaunchSpec(
                cmd=[
                    str(self.venv_python), "-m",
                    "musubi_tuner.kandinsky5_cache_text_encoder_outputs",
                    "--dataset_config", str(flat_toml),
                    "--text_encoder_qwen", self.text_encoder_qwen_path,
                    "--text_encoder_clip", self.text_encoder_clip_path,
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
        if not isinstance(config, Kandinsky5LoRAConfig):
            raise TypeError(f"expected Kandinsky5LoRAConfig, got {type(config).__name__}")
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
            "--task", self.task,
            "--dit", self.dit_path,
            "--vae", self.vae_path,
            "--text_encoder_qwen", self.text_encoder_qwen_path,
            "--text_encoder_clip", self.text_encoder_clip_path,
            "--dataset_config", str(run_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--network_module", "networks.lora_kandinsky",
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
