"""HiDream-O1 LoRA trainer adapter — wraps musubi-tuner hidream_o1_train_network.

HiDream-O1-Image is unusual among the musubi image adapters and intentionally
diverges from the qwen-image / z-image / flux-2 shape:

  - SINGLE model-weight argument ``--dit``. There is NO ``--vae`` and NO
    ``--text_encoder``: HiDream loads its tokenizer/processor/config assets
    automatically from the official HiDream-ai HF repos based on
    ``--model_type``. So this adapter's constructor takes only ``dit_path``.
  - Pre-caching uses HiDream-specific scripts that take NO weight paths:
      pixel cache:  musubi_tuner.hidream_o1_cache_pixel        (no --vae)
      text cache:   musubi_tuner.hidream_o1_cache_text_encoder_outputs (no --text_encoder)
    HiDream caches pixel patch tokens, not VAE latents (it has no VAE latent
    cache in this implementation). We therefore can't reuse the shared
    ``_musubi_pre_cache_commands`` helper (it always passes ``--vae`` /
    ``--text_encoder``) — see ``_hidream_pre_cache_commands`` below.
  - Flow-matching uses uniform timestep sampling + noise-scale params rather
    than ``--discrete_flow_shift``:
      full: --noise_scale_start 8.0 --noise_scale_end 8.0 --noise_clip_std 0.0
      dev:  --noise_scale_start 7.5 --noise_scale_end 7.5 --noise_clip_std 2.5
  - Network module: ``networks.lora_hidream_o1``.

Module names used by musubi-tuner upstream (docs/hidream_o1.md):
  train:       musubi_tuner.hidream_o1_train_network
  cache pixel: musubi_tuner.hidream_o1_cache_pixel
  cache TE:    musubi_tuner.hidream_o1_cache_text_encoder_outputs

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

logger = logging.getLogger("bracket.trainer.hidream_lora")

_MODEL_CLASS = "hidream-o1"

# Per-model_type noise-schedule params (docs/hidream_o1.md). The DiT denoising
# uniform timestep sampling is paired with these noise-scale values; they are
# not loss-bearing search knobs, so we pin them by model_type.
_NOISE_PARAMS_BY_MODEL_TYPE: dict[str, dict[str, float]] = {
    "full": {"noise_scale_start": 8.0, "noise_scale_end": 8.0, "noise_clip_std": 0.0},
    "dev": {"noise_scale_start": 7.5, "noise_scale_end": 7.5, "noise_clip_std": 2.5},
}


@dataclass
class HiDreamLoRAConfig(TrainerConfig):
    learning_rate: float = 4e-5
    optimizer_type: str = "AdamW8bit"
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    network_dim: int = 32
    network_alpha: float = 16.0
    train_batch_size: int = 1            # written into dataset TOML, not CLI
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"
    max_grad_norm: float = 1.0
    model_type: str = "full"             # "full" | "dev" — selects noise params
    task: str = "t2i"                    # "t2i" | "i2i"
    gradient_checkpointing: bool = True
    flash_attn: bool = False
    blocks_to_swap: int = 0
    dataloader_workers: int = 2


class HiDreamLoRATrainer(Trainer):
    name = "hidream-lora"

    def __init__(
        self,
        *,
        musubi_dir: Path,
        venv_python: Path,
        dit_path: str,
        vram_gb: Optional[float] = None,
    ) -> None:
        self.musubi_dir = Path(musubi_dir).resolve()
        self.venv_python = Path(venv_python).resolve()
        self.dit_path = dit_path
        self.train_script = (
            self.musubi_dir / "src" / "musubi_tuner" / "hidream_o1_train_network.py"
        )
        if not self.train_script.exists():
            self.train_script = self.musubi_dir / "hidream_o1_train_network.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"hidream_o1_train_network.py not found under {self.musubi_dir}"
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
            name=f"hidream-lora-v0.1-{self.tier}",
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
                "model_type": FixedKnob(value="full"),
                "task": FixedKnob(value="t2i"),
                "flash_attn": FixedKnob(value=False),
                "blocks_to_swap": FixedKnob(value=BLOCKS_TO_SWAP_BY_TIER[self.tier]),
            },
        )

    def baseline_config(self) -> HiDreamLoRAConfig:
        return HiDreamLoRAConfig(
            learning_rate=4e-5,
            optimizer_type="AdamW8bit",
            lr_scheduler="cosine",
            lr_warmup_steps=100,
            network_dim=32,
            network_alpha=16.0,
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
        # NOTE: every curated LR stays inside the search-space FloatKnob range
        # [1e-6, 2e-4] so curated configs validate against the declared space
        # (Prodigy's usual lr=1.0 multiplier would fall outside that range, so
        # we don't seed a Prodigy curated here — Prodigy is still reachable as
        # an optimizer choice during the search phase).
        return [
            # 1) Documented starter — 4e-5 AdamW8bit is the LR from the official
            #    musubi HiDream-O1 LoRA example.
            HiDreamLoRAConfig(
                learning_rate=4e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 2) Lower LR + warmup — safer if 4e-5 diverges on a dataset.
            HiDreamLoRAConfig(
                learning_rate=1e-5, optimizer_type="AdamW8bit",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=32, network_alpha=16.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
            # 3) Higher-rank adapter at the documented LR — more capacity for
            #    detailed subjects; Lion tends to pair well with HiDream.
            HiDreamLoRAConfig(
                learning_rate=8e-5, optimizer_type="Lion",
                lr_scheduler="cosine", lr_warmup_steps=100,
                network_dim=64, network_alpha=32.0,
                train_batch_size=bs, gradient_checkpointing=ckpt,
                dataloader_workers=workers, blocks_to_swap=swap,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> HiDreamLoRAConfig:
        return HiDreamLoRAConfig(
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
            model_type=str(knobs["model_type"]),
            task=str(knobs["task"]),
            gradient_checkpointing=bool(knobs["gradient_checkpointing"]),
            flash_attn=bool(knobs["flash_attn"]),
            blocks_to_swap=int(knobs["blocks_to_swap"]),
            dataloader_workers=int(knobs["dataloader_workers"]),
        )

    def session_setup_commands(
        self, *, dataset_toml: Path, run_dir: Path,
    ) -> list[LaunchSpec]:
        return _hidream_pre_cache_commands(
            musubi_dir=self.musubi_dir,
            venv_python=self.venv_python,
            run_dir=run_dir,
            dataset_toml=dataset_toml,
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
        if not isinstance(config, HiDreamLoRAConfig):
            raise TypeError(
                f"expected HiDreamLoRAConfig, got {type(config).__name__}"
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

        noise = _NOISE_PARAMS_BY_MODEL_TYPE.get(
            config.model_type, _NOISE_PARAMS_BY_MODEL_TYPE["full"]
        )

        cmd: list[str] = [
            *make_accelerate_launch_prefix(
                self.venv_python, mixed_precision=config.mixed_precision
            ),
            str(self.train_script),
            # HiDream takes a single weight arg — no --vae, no --text_encoder.
            "--dit", self.dit_path,
            "--dataset_config", str(run_toml),
            "--output_dir", str(output_dir),
            "--output_name", "candidate",
            "--logging_dir", str(logging_dir),
            "--log_with", "tensorboard",
            "--log_prefix", "run_",
            "--model_type", config.model_type,
            "--task", config.task,
            "--network_module", "networks.lora_hidream_o1",
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
            # HiDream-O1 uses uniform timestep sampling + noise-scale params,
            # NOT discrete_flow_shift.
            "--timestep_sampling", "uniform",
            "--weighting_scheme", "none",
            "--noise_scale_start", f"{noise['noise_scale_start']:.4f}",
            "--noise_scale_end", f"{noise['noise_scale_end']:.4f}",
            "--noise_clip_std", f"{noise['noise_clip_std']:.4f}",
            "--max_data_loader_n_workers", str(config.dataloader_workers),
            "--persistent_data_loader_workers",
            "--save_every_n_steps",
            str(resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)),
            "--no_metadata",
        ]
        if config.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if config.flash_attn:
            cmd.append("--flash_attn")
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

    @staticmethod
    def pre_caching_hint() -> str:
        """HiDream-O1 needs pixel-patch + text caches BEFORE training. Run once
        per session (or per dataset change) before invoking the orchestrator:

            python -m musubi_tuner.hidream_o1_cache_pixel \\
                --dataset_config <subset/dataset.toml> --batch_size 1
            python -m musubi_tuner.hidream_o1_cache_text_encoder_outputs \\
                --dataset_config <subset/dataset.toml> --batch_size 16

        Unlike the other musubi image trainers, neither cache step takes a
        weight path — HiDream loads tokenizer/processor assets from the
        official HF repos automatically. The orchestrator runs these for you
        via ``session_setup_commands``.
        """
        return HiDreamLoRATrainer.pre_caching_hint.__doc__ or ""


def _hidream_pre_cache_commands(
    *,
    musubi_dir: Path,
    venv_python: Path,
    run_dir: Path,
    dataset_toml: Path,
) -> list[LaunchSpec]:
    """Pre-caching cmd builder for HiDream-O1.

    Distinct from the shared ``_musubi_pre_cache_commands``: HiDream caches
    pixel patch tokens (not VAE latents) and its cache scripts take NO weight
    paths (no --vae, no --text_encoder). Tokenizer/processor assets load from
    the official HF repos automatically. Idempotent — the cache scripts skip
    files that already exist.
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
                str(venv_python), "-m", "musubi_tuner.hidream_o1_cache_pixel",
                "--dataset_config", str(flat_toml),
                "--batch_size", "1",
            ],
            cwd=musubi_dir, env=env,
            output_dir=run_dir, logging_dir=logging_dir,
            tfevents_glob="",
        ),
        LaunchSpec(
            cmd=[
                str(venv_python), "-m",
                "musubi_tuner.hidream_o1_cache_text_encoder_outputs",
                "--dataset_config", str(flat_toml),
                "--batch_size", "16",
            ],
            cwd=musubi_dir, env=env,
            output_dir=run_dir, logging_dir=logging_dir,
            tfevents_glob="",
        ),
    ]
