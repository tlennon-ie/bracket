"""Model + training-type preset registry.

Drives the UI's cascading dropdowns and the orchestrator's trainer
construction. Each preset declares:

  - which Trainer adapter class to instantiate
  - which constructor kwargs to map from UI fields
  - what each UI field is (label, default value, required vs optional, kind)
  - any run-time notes (e.g. "Flux-2 LoRA needs Mistral-3-Small TE")

Adding a new preset = add an entry to PRESETS. UI rebuilds automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from bracket.trainer.base import Trainer


# Default paths are resolved (in priority order):
#   1. Environment variables (BRACKET_VENV_PYTHON, BRACKET_MUSUBI_DIR, …)
#   2. The bundled trainers root at ~/.cache/bracket/trainers/ (created by
#      install.sh / install.ps1 / install.bat — see scripts/install_*).
#   3. An empty string — the user fills it in via the Setup tab.
#
# Nothing in this module assumes a specific developer's filesystem. Public
# users land on empty defaults if they haven't run the installer; pre-existing
# users can keep their setup by exporting the env vars.

_TRAINERS_ROOT = Path(
    os.environ.get("BRACKET_TRAINERS_ROOT", str(Path.home() / ".cache" / "bracket" / "trainers"))
)


def _env_or_default(env_var: str, default: str) -> str:
    """Return the env-var value if set, else the default. Empty string if neither."""
    return os.environ.get(env_var, default)


def _venv_python_default() -> str:
    """Resolve the trainer venv python — env var or installed location."""
    if (env := os.environ.get("BRACKET_VENV_PYTHON")):
        return env
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    candidate = _TRAINERS_ROOT / "venv" / suffix
    return str(candidate) if candidate.exists() else ""


def _musubi_dir_default() -> str:
    if (env := os.environ.get("BRACKET_MUSUBI_DIR")):
        return env
    candidate = _TRAINERS_ROOT / "musubi-tuner"
    return str(candidate) if candidate.exists() else ""


def _sd_scripts_default() -> str:
    if (env := os.environ.get("BRACKET_SD_SCRIPTS_DIR")):
        return env
    candidate = _TRAINERS_ROOT / "musubi-tuner" / "sd-scripts"
    return str(candidate) if candidate.exists() else ""


_DEFAULT_VENV_PYTHON = _venv_python_default()
_DEFAULT_SD_SCRIPTS = _sd_scripts_default()
_DEFAULT_MUSUBI_DIR = _musubi_dir_default()
# These are example paths only — public users will set them via Setup tab
# once they've downloaded the weights they want to fine-tune.
_DEFAULT_SDXL_PRETRAINED = _env_or_default("BRACKET_SDXL_PRETRAINED", "")
_DEFAULT_DATASET_TOML = _env_or_default("BRACKET_DATASET_TOML", "")
_DEFAULT_SAMPLE_PROMPTS = _env_or_default("BRACKET_SAMPLE_PROMPTS", "")
_DEFAULT_VAE_PATH = _env_or_default("BRACKET_VAE_PATH", "")
_DEFAULT_QWEN3_TE_PATH = _env_or_default("BRACKET_QWEN3_TE_PATH", "")
_DEFAULT_FLUX2_DIT_PATH = _env_or_default("BRACKET_FLUX2_DIT_PATH", "")
_DEFAULT_MISTRAL3_TE_PATH = _env_or_default("BRACKET_MISTRAL3_TE_PATH", "")


@dataclass(frozen=True)
class FieldSpec:
    """One UI field — feeds into the trainer kwargs OR the orchestrator config."""
    name: str             # internal key (also the trainer ctor kwarg name when target == "trainer")
    label: str            # UI label
    default: str = ""     # default value displayed
    required: bool = True
    kind: str = "string"  # "string" | "path" (file) | "dir" (directory) | "filepath_optional"
    help: str = ""        # info-tooltip text
    target: str = "trainer"  # "trainer" → ctor kwarg | "session" → orchestrator/run param


# Reusable field definitions for cross-preset sharing.
_VENV_PYTHON_FIELD = FieldSpec(
    name="venv_python", label="Trainer venv python.exe *", default=_DEFAULT_VENV_PYTHON,
    required=True, kind="path", help="The Python executable inside the trainer's venv.",
)


@dataclass(frozen=True)
class ModelPreset:
    id: str                     # unique key, e.g. "sdxl-lora"
    model_family: str           # "SDXL", "Z-Image", "Flux-2-Klein"
    training_type: str          # "LoRA" or "Full FT"
    display_name: str           # "SDXL · LoRA"
    trainer_factory: Callable[..., Trainer]   # called with kwargs from FieldSpec(target='trainer')
    fields: tuple[FieldSpec, ...] = ()
    notes: str = ""             # markdown shown in UI
    needs_pre_cache: bool = False  # informational; the trainer's own session_setup_commands does the work


# Each entry below is one (model, training-type) we explicitly support today.
# Models with no full-FT script in their upstream (e.g. Flux-2-Klein in musubi)
# simply don't have a Full-FT preset.

def _build_sdxl_lora(**kw: Any) -> Trainer:
    from bracket.trainer.sdxl import SDXLTrainer
    return SDXLTrainer(
        sd_scripts_dir=kw["sd_scripts_dir"],
        venv_python=kw["venv_python"],
        pretrained_model=kw["pretrained_model"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_sdxl_full(**kw: Any) -> Trainer:
    from bracket.trainer.sdxl_full import SDXLFullTrainer
    return SDXLFullTrainer(
        sd_scripts_dir=kw["sd_scripts_dir"],
        venv_python=kw["venv_python"],
        pretrained_model=kw["pretrained_model"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_zimage_lora(**kw: Any) -> Trainer:
    from bracket.trainer.zimage_lora import ZImageLoRATrainer
    return ZImageLoRATrainer(
        musubi_dir=kw["musubi_dir"],
        venv_python=kw["venv_python"],
        dit_path=kw["dit_path"],
        vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_zimage_full(**kw: Any) -> Trainer:
    from bracket.trainer.zimage_full import ZImageFullTrainer
    return ZImageFullTrainer(
        musubi_dir=kw["musubi_dir"],
        venv_python=kw["venv_python"],
        dit_path=kw["dit_path"],
        vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_flux2_klein_lora(**kw: Any) -> Trainer:
    from bracket.trainer.flux2_klein_lora import Flux2KleinLoRATrainer
    return Flux2KleinLoRATrainer(
        musubi_dir=kw["musubi_dir"],
        venv_python=kw["venv_python"],
        dit_path=kw["dit_path"],
        vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


_SDXL_PRETRAINED_FIELD = FieldSpec(
    name="pretrained_model", label="SDXL base path *",
    default=_DEFAULT_SDXL_PRETRAINED, required=True, kind="path",
    help="HF format directory OR single .safetensors. The SDXL base 1.0 release.",
)
_SD_SCRIPTS_FIELD = FieldSpec(
    name="sd_scripts_dir", label="sd-scripts directory *",
    default=_DEFAULT_SD_SCRIPTS, required=True, kind="dir",
    help="Path to the sd-scripts (Kohya) checkout. Bundled inside musubi-tuner.",
)

_MUSUBI_DIR_FIELD = FieldSpec(
    name="musubi_dir", label="musubi-tuner directory *",
    default=_DEFAULT_MUSUBI_DIR, required=True, kind="dir",
)
_DIT_FIELD = FieldSpec(
    name="dit_path", label="DiT weights (.safetensors) *",
    default="", required=True, kind="path",
)
_VAE_FIELD = FieldSpec(
    name="vae_path", label="VAE weights (.safetensors) *",
    default=_DEFAULT_VAE_PATH, required=True, kind="path",
    help="Path to the VAE checkpoint (.safetensors). Set BRACKET_VAE_PATH to change the default.",
)


PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        id="sdxl-lora",
        model_family="SDXL",
        training_type="LoRA",
        display_name="SDXL · LoRA",
        trainer_factory=_build_sdxl_lora,
        fields=(
            _SDXL_PRETRAINED_FIELD,
            _SD_SCRIPTS_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `sd-scripts/sdxl_train_network.py`. Caches latents + text-encoder "
            "outputs on the fly inside the trainer — no manual pre-cache needed."
        ),
        needs_pre_cache=False,
    ),
    ModelPreset(
        id="sdxl-full",
        model_family="SDXL",
        training_type="Full FT",
        display_name="SDXL · Full FT",
        trainer_factory=_build_sdxl_full,
        fields=(
            _SDXL_PRETRAINED_FIELD,
            _SD_SCRIPTS_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `sd-scripts/sdxl_train.py`. Adafactor + fused backward pass. "
            "Higher VRAM than LoRA — orchestrator's batch-size search starts smaller."
        ),
        needs_pre_cache=False,
    ),
    ModelPreset(
        id="zimage-lora",
        model_family="Z-Image",
        training_type="LoRA",
        display_name="Z-Image · LoRA",
        trainer_factory=_build_zimage_lora,
        fields=(
            _DIT_FIELD,
            _VAE_FIELD,
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Qwen3) *",
                default=_DEFAULT_QWEN3_TE_PATH,
                required=True, kind="path",
                help="Set BRACKET_QWEN3_TE_PATH to change the default.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/zimage_train_network.py`. **Latent + TE caching is "
            "run automatically once per session** before the first candidate. "
            "Z-Image uses Qwen3 as the text encoder."
        ),
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="zimage-full",
        model_family="Z-Image",
        training_type="Full FT",
        display_name="Z-Image · Full FT",
        trainer_factory=_build_zimage_full,
        fields=(
            _DIT_FIELD,
            _VAE_FIELD,
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Qwen3) *",
                default=_DEFAULT_QWEN3_TE_PATH,
                required=True, kind="path",
                help="Set BRACKET_QWEN3_TE_PATH to change the default.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/zimage_train.py`. Full fine-tune with Adafactor + "
            "fused backward + full_bf16. Pre-cache runs automatically."
        ),
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="flux2-klein-lora",
        model_family="Flux-2-Klein",
        training_type="LoRA",
        display_name="Flux-2-Klein 9B · LoRA",
        trainer_factory=_build_flux2_klein_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights (.safetensors) *",
                default=_DEFAULT_FLUX2_DIT_PATH,
                required=True, kind="path",
                help="Set BRACKET_FLUX2_DIT_PATH to change the default.",
            ),
            _VAE_FIELD,
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Mistral-3-Small) *",
                default=_DEFAULT_MISTRAL3_TE_PATH,
                required=True, kind="path",
                help="Set BRACKET_MISTRAL3_TE_PATH to change the default.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/flux_2_train_network.py`. Flux-2-Klein 9B fp8 "
            "is much smaller than the 32B flagship — comfortably fits 5090 VRAM. "
            "Mistral-3-Small as the text encoder. Pre-cache runs automatically."
        ),
        needs_pre_cache=True,
    ),
)


# Always-shown session/orchestration fields (independent of preset).
SESSION_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="dataset_toml", label="Dataset config TOML *",
        default=_DEFAULT_DATASET_TOML, required=True, kind="path",
        help="sd-scripts / musubi-tuner dataset config TOML.",
        target="session",
    ),
    FieldSpec(
        name="output_dir", label="Session output directory *",
        default="./runs/session-001", required=True, kind="dir",
        help="Where ledger, runs, samples, and the proof report land.",
        target="session",
    ),
    FieldSpec(
        name="sample_prompts", label="Sample prompts file (optional)",
        default=_DEFAULT_SAMPLE_PROMPTS, required=False, kind="filepath_optional",
        help="Triggers sample image generation per run AND enables the VLM judge.",
        target="session",
    ),
    FieldSpec(
        name="resume", label="Resume from prior LoRA / state (optional)",
        default="", required=False, kind="filepath_optional",
        help="Path to a previously-trained LoRA or training state to warm-start from.",
        target="session",
    ),
)


def list_model_families() -> list[str]:
    """Unique families, in declaration order."""
    seen: list[str] = []
    for p in PRESETS:
        if p.model_family not in seen:
            seen.append(p.model_family)
    return seen


def training_types_for(family: str) -> list[str]:
    return [p.training_type for p in PRESETS if p.model_family == family]


def get_preset(family: str, training_type: str) -> Optional[ModelPreset]:
    for p in PRESETS:
        if p.model_family == family and p.training_type == training_type:
            return p
    return None
