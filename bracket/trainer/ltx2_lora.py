"""LTX-2 LoRA — wraps Lightricks' native ``ltx-trainer`` (T2V + I2V).

Unlike every other adapter in Bracket, ltx-trainer is driven by a single
**YAML config file**, not CLI flags. :meth:`prepare_run` WRITES a
Pydantic-validated ``config.yaml`` and returns a :class:`LaunchSpec` that runs

    uv run python scripts/train.py <ABSOLUTE config.yaml> --disable-progress-bars

The ``--disable-progress-bars`` flag is required: it makes the trainer emit the
newline-terminated ``Step N/M - Loss: ...`` cadence lines that Bracket's
``bracket.log_loss_reader`` parses. LTX-2 writes no tfevents, so the returned
LaunchSpec sets ``loss_source="stdout_log"`` — the runner skips the tfevents
glob and the scorer reads loss from the run's stdout log instead.

Preprocessing (``scripts/process_dataset.py``) runs ONCE per session via
:meth:`session_setup_commands`, writing ``latents/`` / ``conditions/`` /
``audio_latents/`` under a DETERMINISTIC root derived from the source dataset
(``<dataset_toml.parent>/.precomputed_ltx2``) so every candidate reuses it.

``mode`` selects the training variant:
  - ``"t2v"`` — text-to-video (no extra conditioning).
  - ``"i2v"`` — image-to-video; adds a ``first_frame`` condition to the
    flexible training strategy's video stream.

LTX-2 is a joint audio-video model; the config enables the audio latent stream
and audio generation in validation. The text encoder is Gemma.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import toml
import yaml

from bracket.dataset.ltx2_dataset import (
    build_ltx2_dataset_json, preprocessed_root_for,
)
from bracket.judge.base import parse_judge_prompts_file
from bracket.search.space import (
    CategoricalKnob, FixedKnob, FloatKnob, SearchSpace,
)
from bracket.trainer.base import (
    LaunchSpec, Trainer, TrainerConfig,
    make_subprocess_env, resolve_save_every_n_steps,
)

logger = logging.getLogger("bracket.trainer.ltx2_lora")

_VALID_MODES = ("t2v", "i2v")
# Default resolution bucket when the source TOML declares no resolution.
# Frame count is pinned to a multiple LTX-2's temporal compression accepts.
_DEFAULT_BUCKET = "768x768x49"
_BUCKET_FRAMES = 49
_LORA_TARGET_MODULES = ("to_k", "to_q", "to_v", "to_out.0")
_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


@dataclass
class LTX2LoRAConfig(TrainerConfig):
    learning_rate: float = 1e-4
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision_mode: str = "bf16"
    enable_gradient_checkpointing: bool = True
    num_dataloader_workers: int = 2


def _resolution_buckets_from_toml(source_toml: Path) -> str:
    """Derive a ``WxHxF`` resolution-bucket string from the source TOML.

    Reads the dataset ``resolution`` ([general] or [[datasets]] level; list
    ``[w, h]`` or scalar ``s``). Frame count is pinned to ``_BUCKET_FRAMES``.
    Falls back to ``_DEFAULT_BUCKET`` when no resolution is declared.
    """
    try:
        data = toml.load(Path(source_toml))
    except (OSError, toml.TomlDecodeError):
        return _DEFAULT_BUCKET
    res: Any = data.get("general", {}).get("resolution")
    if res is None:
        for ds in data.get("datasets", []):
            if "resolution" in ds:
                res = ds["resolution"]
                break
    if res is None:
        return _DEFAULT_BUCKET
    if isinstance(res, (list, tuple)) and len(res) >= 2:
        w, h = int(res[0]), int(res[1])
    elif isinstance(res, (int, float)):
        w = h = int(res)
    else:
        return _DEFAULT_BUCKET
    return f"{w}x{h}x{_BUCKET_FRAMES}"


class LTX2LoRATrainer(Trainer):
    """Adapter for Lightricks' native ltx-trainer (LTX-2), LoRA only."""

    def __init__(
        self,
        *,
        trainer_dir: Path,
        model_path: str,
        text_encoder_path: str,
        mode: str = "t2v",
        vram_gb: Optional[float] = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.mode = mode
        self.trainer_dir = Path(trainer_dir).resolve()
        self.model_path = str(model_path)
        self.text_encoder_path = str(text_encoder_path)
        self.train_script = self.trainer_dir / "scripts" / "train.py"
        if not self.train_script.exists():
            raise FileNotFoundError(
                f"scripts/train.py not found under {self.trainer_dir}"
            )
        self.process_script = self.trainer_dir / "scripts" / "process_dataset.py"
        self.vram_gb = float(vram_gb) if vram_gb is not None else None
        self.name = f"ltx2-lora-{mode}"

    # ─────────────────────────── search space ───────────────────────────

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(
            name=f"ltx2-lora-{self.mode}-v0.1",
            knobs={
                "learning_rate": FloatKnob(low=1e-5, high=5e-4, log=True),
                "lora_rank": CategoricalKnob(choices=(16, 32, 64)),
                "lora_alpha": CategoricalKnob(choices=(16, 32, 64)),
                "scheduler_type": CategoricalKnob(
                    choices=("constant", "linear", "cosine", "cosine_with_restarts"),
                ),
                "optimizer_type": CategoricalKnob(choices=("adamw", "adamw8bit")),
                "lora_dropout": FixedKnob(value=0.0),
                "batch_size": FixedKnob(value=1),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "max_grad_norm": FixedKnob(value=1.0),
                "mixed_precision_mode": FixedKnob(value="bf16"),
                "enable_gradient_checkpointing": FixedKnob(value=True),
                "num_dataloader_workers": FixedKnob(value=2),
            },
        )

    def baseline_config(self) -> LTX2LoRAConfig:
        return LTX2LoRAConfig(
            learning_rate=1e-4, lora_rank=32, lora_alpha=32,
            optimizer_type="adamw", scheduler_type="linear",
        )

    def curated_configs(self) -> list[TrainerConfig]:
        return [
            LTX2LoRAConfig(
                learning_rate=1e-4, lora_rank=32, lora_alpha=32,
                optimizer_type="adamw", scheduler_type="cosine",
            ),
            LTX2LoRAConfig(
                learning_rate=5e-5, lora_rank=16, lora_alpha=16,
                optimizer_type="adamw", scheduler_type="linear",
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> LTX2LoRAConfig:
        return LTX2LoRAConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer_type=str(knobs["optimizer_type"]),
            scheduler_type=str(knobs["scheduler_type"]),
            lora_rank=int(knobs["lora_rank"]),
            lora_alpha=int(knobs["lora_alpha"]),
            lora_dropout=float(knobs["lora_dropout"]),
            batch_size=int(knobs["batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            max_grad_norm=float(knobs["max_grad_norm"]),
            mixed_precision_mode=str(knobs["mixed_precision_mode"]),
            enable_gradient_checkpointing=bool(knobs["enable_gradient_checkpointing"]),
            num_dataloader_workers=int(knobs["num_dataloader_workers"]),
        )

    # ─────────────────────────── session setup ───────────────────────────

    def session_setup_commands(
        self, *, dataset_toml: Path, run_dir: Path,
    ) -> list[LaunchSpec]:
        """Build the dataset.json bridge + the once-per-session preprocessing.

        The dataset.json is written under ``run_dir`` (per the contract), but
        the preprocessing OUTPUT root is the deterministic
        ``<dataset_toml.parent>/.precomputed_ltx2`` so every candidate reuses
        the single pass.
        """
        run_dir = Path(run_dir).resolve()
        logging_dir = run_dir / "logs"
        logging_dir.mkdir(parents=True, exist_ok=True)

        dataset_json = build_ltx2_dataset_json(
            source_toml=Path(dataset_toml),
            target_json=run_dir / "ltx2_dataset.json",
        )
        preprocessed_root = preprocessed_root_for(Path(dataset_toml))
        preprocessed_root.mkdir(parents=True, exist_ok=True)
        buckets = _resolution_buckets_from_toml(Path(dataset_toml))

        return [
            LaunchSpec(
                cmd=[
                    "uv", "run", "python", "scripts/process_dataset.py",
                    str(dataset_json),
                    "--resolution-buckets", buckets,
                    "--model-path", self.model_path,
                    "--text-encoder-path", self.text_encoder_path,
                    "--output-dir", str(preprocessed_root),
                ],
                cwd=self.trainer_dir, env=make_subprocess_env(),
                output_dir=run_dir, logging_dir=logging_dir,
                tfevents_glob="",
                # Preprocessing isn't loss-scored, but mark it stdout_log for
                # consistency so the runner never globs for (absent) tfevents.
                loss_source="stdout_log",
            ),
        ]

    # ─────────────────────────── prepare_run ───────────────────────────

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
        if not isinstance(config, LTX2LoRAConfig):
            raise TypeError(f"expected LTX2LoRAConfig, got {type(config).__name__}")

        run_dir = Path(run_dir).resolve()
        output_dir = run_dir / "output"
        logging_dir = run_dir / "logs"
        sample_dir = output_dir / "samples"
        for d in (output_dir, logging_dir, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        preprocessed_root = preprocessed_root_for(Path(dataset_toml))
        config_yaml = run_dir / "config.yaml"

        document = self._build_config_document(
            config=config,
            output_dir=output_dir,
            preprocessed_root=preprocessed_root,
            max_steps=max_steps,
            seed=seed,
            sample_prompts=sample_prompts,
            sample_every_n_steps=sample_every_n_steps,
            save_every_n_steps=save_every_n_steps,
            save_state=save_state,
            resume_from=resume_from,
        )
        config_yaml.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8",
        )
        logger.info("wrote LTX-2 training config: %s", config_yaml)

        return LaunchSpec(
            cmd=[
                "uv", "run", "python", "scripts/train.py",
                str(config_yaml), "--disable-progress-bars",
            ],
            cwd=self.trainer_dir, env=make_subprocess_env(),
            output_dir=output_dir, logging_dir=logging_dir,
            tfevents_glob="", sample_dir=sample_dir,
            loss_source="stdout_log",
        )

    # ─────────────────────────── YAML builder ───────────────────────────

    def _build_config_document(
        self,
        *,
        config: LTX2LoRAConfig,
        output_dir: Path,
        preprocessed_root: Path,
        max_steps: int,
        seed: int,
        sample_prompts: Optional[Path],
        sample_every_n_steps: Optional[int],
        save_every_n_steps: Optional[int],
        save_state: bool,
        resume_from: Optional[Path],
    ) -> dict[str, Any]:
        """Assemble the Pydantic-validated ltx-trainer config dict."""
        video_strategy: dict[str, Any] = {
            "is_generated": True, "latents_dir": "latents",
        }
        if self.mode == "i2v":
            video_strategy["conditions"] = [
                {"type": "first_frame", "probability": 0.5},
            ]

        samples = [
            {"prompt": p}
            for p in (parse_judge_prompts_file(sample_prompts) if sample_prompts else [])
        ]
        validation_interval = (
            int(sample_every_n_steps)
            if sample_prompts is not None and sample_every_n_steps
            else None
        )

        checkpoint_interval = (
            resolve_save_every_n_steps(save_every_n_steps, max_steps=max_steps)
            if save_every_n_steps and save_every_n_steps > 0
            else None
        )

        checkpoints: dict[str, Any] = {
            "interval": checkpoint_interval,
            "keep_last_n": -1,
            "precision": "bfloat16",
            "save_training_state": "full" if save_state else "minimal",
        }

        return {
            "seed": int(seed),
            "output_dir": str(output_dir),
            "model": {
                "model_path": self.model_path,
                "text_encoder_path": self.text_encoder_path,
                "training_mode": "lora",
                "load_checkpoint": str(resume_from) if resume_from is not None else None,
            },
            "lora": {
                "rank": int(config.lora_rank),
                "alpha": int(config.lora_alpha),
                "dropout": float(config.lora_dropout),
                "target_modules": list(_LORA_TARGET_MODULES),
            },
            "training_strategy": {
                "name": "flexible",
                "video": video_strategy,
                "audio": {"is_generated": True, "latents_dir": "audio_latents"},
            },
            "optimization": {
                "learning_rate": float(config.learning_rate),
                "steps": int(max_steps),
                "batch_size": int(config.batch_size),
                "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
                "max_grad_norm": float(config.max_grad_norm),
                "optimizer_type": config.optimizer_type,
                "scheduler_type": config.scheduler_type,
                "enable_gradient_checkpointing": bool(config.enable_gradient_checkpointing),
            },
            "acceleration": {
                "mixed_precision_mode": config.mixed_precision_mode,
                "quantization": None,
            },
            "data": {
                "preprocessed_data_root": str(preprocessed_root),
                "num_dataloader_workers": int(config.num_dataloader_workers),
            },
            "validation": {
                "samples": samples,
                "negative_prompt": _NEGATIVE_PROMPT,
                "video_dims": [576, 576, 89],
                "frame_rate": 25.0,
                "seed": int(seed),
                "inference_steps": 30,
                "interval": validation_interval,
                "guidance_scale": 4.0,
                "generate_audio": True,
            },
            "checkpoints": checkpoints,
            "flow_matching": {"timestep_sampling_mode": "shifted_logit_normal"},
            "wandb": {"enabled": False},
            "hub": {"push_to_hub": False},
        }
