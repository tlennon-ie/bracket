"""ai-toolkit LoRA — wraps ostris' native ``ai-toolkit`` (``sd_trainer`` job).

Like LTX-2's ltx-trainer, ai-toolkit is driven by a single **YAML config
file**, not CLI flags. :meth:`prepare_run` WRITES a ``config.yaml`` and returns
a :class:`LaunchSpec` that runs::

    <aitk_venv_python> run.py <ABSOLUTE config.yaml>

with ``cwd`` set to the ai-toolkit checkout. No ``accelerate`` wrapper is
needed — the device is pinned in the config (``device: cuda:0``). There is NO
separate preprocessing step: latents cache inline
(``cache_latents_to_disk: true``), so this adapter declares no
``session_setup_commands``.

ai-toolkit writes loss to a structured SQLite database (``loss_log.db``) under
the save root when ``logging.use_ui_logger: true`` — Bracket's scorer reads
that DB. The returned LaunchSpec therefore sets ``loss_source="sqlite_db"`` and
``loss_db_path`` (the runner skips the tfevents glob; the scorer parses the DB).

Derived save layout (ai-toolkit):
    training_folder = ``<run_dir>/output``
    save_root       = ``<training_folder>/<name>`` = ``<run_dir>/output/candidate``
    loss DB         = ``<save_root>/loss_log.db``
    samples         = ``<save_root>/samples``

``model_id`` + ``model_extra`` parametrise the model (chroma, lumina2,
omnigen2, flex1, flex2). ``model_extra`` carries the architecture SELECTOR and
quantize mode verbatim from each model's ai-toolkit example config — these
differ per model (``arch: chroma`` vs ``is_lumina2: true`` vs ``is_flux: true``;
``quantize`` vs ``quantize_te``). ``bypass_guidance_embedding`` is set for the
Flex models, which require it during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from bracket.dataset.aitk_dataset import image_dir_for
from bracket.judge.base import parse_judge_prompts_file
from bracket.search.space import (
    CategoricalKnob, FixedKnob, FloatKnob, SearchSpace,
)
from bracket.trainer.base import (
    LaunchSpec, Trainer, TrainerConfig,
    make_subprocess_env, resolve_save_every_n_steps,
)

logger = logging.getLogger("bracket.trainer.aitk_lora")

# The ai-toolkit process config "name" — also the save-root subdir under
# training_folder. Kept constant so the derived loss_db_path / sample_dir are
# deterministic.
_CANDIDATE_NAME = "candidate"

# Model ids that require bypassing the guidance embedder during training
# (the Flex architectures).
_BYPASS_GUIDANCE_IDS = ("flex1", "flex2")

# Multi-resolution bucket list ai-toolkit examples ship with for every model.
_RESOLUTION = (512, 768, 1024)

# Sampling defaults (mirrors ai-toolkit example configs).
_SAMPLE_WIDTH = 1024
_SAMPLE_HEIGHT = 1024
_SAMPLE_STEPS = 20
_GUIDANCE_SCALE = 4


@dataclass
class AiToolkitLoRAConfig(TrainerConfig):
    """Searchable + pinned knobs for an ai-toolkit LoRA run.

    Searched: ``learning_rate``, ``optimizer``, ``network_rank``,
    ``network_alpha``. Everything else is pinned to the values ai-toolkit's
    example configs ship with.
    """

    learning_rate: float = 1e-4
    optimizer: str = "adamw8bit"
    network_rank: int = 16
    network_alpha: int = 16
    # Pinned — documented here so to_dict()/config_from_dict() roundtrip.
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dtype: str = "bf16"
    gradient_checkpointing: bool = True
    noise_scheduler: str = "flowmatch"


class AiToolkitLoRATrainer(Trainer):
    """Adapter for ostris' native ai-toolkit (``sd_trainer`` job), LoRA only."""

    def __init__(
        self,
        *,
        aitk_dir: Path,
        venv_python: Path,
        model_name_or_path: str,
        model_id: str,
        model_extra: Mapping[str, Any],
        vram_gb: Optional[float] = None,
    ) -> None:
        self.aitk_dir = Path(aitk_dir).resolve()
        self.venv_python = str(venv_python)
        self.model_name_or_path = str(model_name_or_path)
        # Short label used for the trainer/search-space name and the
        # bypass-guidance check (e.g. "chroma", "lumina2", "flex1").
        self.model_id = str(model_id)
        # The model-block keys (beyond ``name_or_path``) verbatim from this
        # model's ai-toolkit example config — the architecture SELECTOR and
        # quantize mode differ per model: chroma/omnigen2/flex2 use
        # ``arch: <name>``; lumina2 uses ``is_lumina2: true``; flex1 uses
        # ``is_flux: true``. quantize vs quantize_te also varies. Merged into
        # the ``model:`` block as-is.
        self.model_extra = dict(model_extra)
        self.run_script = self.aitk_dir / "run.py"
        if not self.run_script.exists():
            raise FileNotFoundError(f"run.py not found under {self.aitk_dir}")
        self.vram_gb = float(vram_gb) if vram_gb is not None else None
        self.name = f"aitk-lora-{self.model_id}"

    # ─────────────────────────── search space ───────────────────────────

    def declare_search_space(self) -> SearchSpace:
        return SearchSpace(
            name=f"aitk-lora-{self.model_id}-v0.1",
            knobs={
                "learning_rate": FloatKnob(low=1e-5, high=5e-4, log=True),
                "network_rank": CategoricalKnob(choices=(8, 16, 32, 64)),
                "network_alpha": CategoricalKnob(choices=(8, 16, 32)),
                "optimizer": CategoricalKnob(choices=("adamw8bit", "adamw")),
                "batch_size": FixedKnob(value=1),
                "gradient_accumulation_steps": FixedKnob(value=1),
                "dtype": FixedKnob(value="bf16"),
                "gradient_checkpointing": FixedKnob(value=True),
                "noise_scheduler": FixedKnob(value="flowmatch"),
            },
        )

    def baseline_config(self) -> AiToolkitLoRAConfig:
        return AiToolkitLoRAConfig(
            learning_rate=1e-4, optimizer="adamw8bit",
            network_rank=16, network_alpha=16,
        )

    def curated_configs(self) -> list[TrainerConfig]:
        return [
            AiToolkitLoRAConfig(
                learning_rate=1e-4, optimizer="adamw8bit",
                network_rank=32, network_alpha=32,
            ),
            AiToolkitLoRAConfig(
                learning_rate=5e-5, optimizer="adamw",
                network_rank=16, network_alpha=16,
            ),
        ]

    def config_from_dict(self, knobs: Mapping[str, Any]) -> AiToolkitLoRAConfig:
        return AiToolkitLoRAConfig(
            learning_rate=float(knobs["learning_rate"]),
            optimizer=str(knobs["optimizer"]),
            network_rank=int(knobs["network_rank"]),
            network_alpha=int(knobs["network_alpha"]),
            batch_size=int(knobs["batch_size"]),
            gradient_accumulation_steps=int(knobs["gradient_accumulation_steps"]),
            dtype=str(knobs["dtype"]),
            gradient_checkpointing=bool(knobs["gradient_checkpointing"]),
            noise_scheduler=str(knobs["noise_scheduler"]),
        )

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
        if not isinstance(config, AiToolkitLoRAConfig):
            raise TypeError(
                f"expected AiToolkitLoRAConfig, got {type(config).__name__}"
            )

        run_dir = Path(run_dir).resolve()
        training_folder = run_dir / "output"
        logging_dir = run_dir / "logs"
        # ai-toolkit save root = <training_folder>/<name>.
        save_root = training_folder / _CANDIDATE_NAME
        sample_dir = save_root / "samples"
        for d in (training_folder, logging_dir, save_root, sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        folder_path, caption_ext = image_dir_for(Path(dataset_toml))
        config_yaml = run_dir / "config.yaml"

        document = self._build_config_document(
            config=config,
            training_folder=training_folder,
            folder_path=folder_path,
            caption_ext=caption_ext,
            max_steps=max_steps,
            seed=seed,
            sample_prompts=sample_prompts,
            sample_every_n_steps=sample_every_n_steps,
            save_every_n_steps=save_every_n_steps,
        )
        config_yaml.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8",
        )
        logger.info("wrote ai-toolkit training config: %s", config_yaml)

        return LaunchSpec(
            cmd=[self.venv_python, "run.py", str(config_yaml)],
            cwd=self.aitk_dir, env=make_subprocess_env(),
            output_dir=training_folder, logging_dir=logging_dir,
            tfevents_glob="", sample_dir=sample_dir,
            loss_source="sqlite_db",
            loss_db_path=save_root / "loss_log.db",
        )

    # ─────────────────────────── YAML builder ───────────────────────────

    def _build_config_document(
        self,
        *,
        config: AiToolkitLoRAConfig,
        training_folder: Path,
        folder_path: str,
        caption_ext: str,
        max_steps: int,
        seed: int,
        sample_prompts: Optional[Path],
        sample_every_n_steps: Optional[int],
        save_every_n_steps: Optional[int],
    ) -> dict[str, Any]:
        """Assemble the ai-toolkit ``sd_trainer`` job config dict."""
        prompts = (
            parse_judge_prompts_file(sample_prompts) if sample_prompts else []
        )
        save_every = resolve_save_every_n_steps(
            save_every_n_steps, max_steps=max_steps,
        )
        sample_every = (
            int(sample_every_n_steps)
            if sample_prompts is not None and sample_every_n_steps
            else max(1, int(max_steps))
        )

        train_block: dict[str, Any] = {
            "batch_size": int(config.batch_size),
            "steps": int(max_steps),
            "gradient_accumulation_steps": int(config.gradient_accumulation_steps),
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": bool(config.gradient_checkpointing),
            "noise_scheduler": config.noise_scheduler,
            "optimizer": config.optimizer,
            "lr": float(config.learning_rate),
            "dtype": config.dtype,
        }
        if self.model_id in _BYPASS_GUIDANCE_IDS:
            # Flex architectures require bypassing the guidance embedder.
            train_block["bypass_guidance_embedding"] = True

        process: dict[str, Any] = {
            "type": "sd_trainer",
            "training_folder": str(training_folder),
            "device": "cuda:0",
            "network": {
                "type": "lora",
                "linear": int(config.network_rank),
                "linear_alpha": int(config.network_alpha),
            },
            "save": {
                "dtype": "float16",
                "save_every": int(save_every),
                "max_step_saves_to_keep": 4,
                "push_to_hub": False,
            },
            "datasets": [
                {
                    "folder_path": folder_path,
                    "caption_ext": caption_ext,
                    "cache_latents_to_disk": True,
                    "resolution": list(_RESOLUTION),
                },
            ],
            "train": train_block,
            "model": {
                "name_or_path": self.model_name_or_path,
                # Per-model selector + quantize keys (arch / is_lumina2 /
                # is_flux, quantize / quantize_te) verbatim from the model's
                # ai-toolkit example config.
                **self.model_extra,
            },
            "sample": {
                "sampler": "flowmatch",
                "sample_every": int(sample_every),
                "width": _SAMPLE_WIDTH,
                "height": _SAMPLE_HEIGHT,
                "prompts": list(prompts),
                "neg": "",
                "seed": int(seed),
                "walk_seed": True,
                "guidance_scale": _GUIDANCE_SCALE,
                "sample_steps": _SAMPLE_STEPS,
            },
            # CRITICAL — enables the SQLite loss_log.db the scorer reads.
            "logging": {"use_ui_logger": True},
        }

        return {
            "job": "extension",
            "config": {
                "name": _CANDIDATE_NAME,
                "process": [process],
            },
            "meta": {"name": _CANDIDATE_NAME, "version": "1.0"},
        }
