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
# Flux.1 (dev/schnell + Kontext) — dual TE: T5-XXL + CLIP-L
_DEFAULT_FLUX1_DIT_PATH = _env_or_default("BRACKET_FLUX1_DIT_PATH", "")
_DEFAULT_FLUX1_AE_PATH = _env_or_default("BRACKET_FLUX1_AE_PATH", "")
_DEFAULT_FLUX1_KONTEXT_DIT_PATH = _env_or_default("BRACKET_FLUX1_KONTEXT_DIT_PATH", "")
_DEFAULT_T5XXL_PATH = _env_or_default("BRACKET_T5XXL_PATH", "")
_DEFAULT_CLIP_L_PATH = _env_or_default("BRACKET_CLIP_L_PATH", "")
# Qwen-Image (text encoder is Qwen2.5-VL-7B; distinct from the Qwen3 used by Z-Image)
_DEFAULT_QWEN_IMAGE_DIT_PATH = _env_or_default("BRACKET_QWEN_IMAGE_DIT_PATH", "")
_DEFAULT_QWEN_IMAGE_VAE_PATH = _env_or_default("BRACKET_QWEN_IMAGE_VAE_PATH", "")
_DEFAULT_QWEN_IMAGE_TE_PATH = _env_or_default("BRACKET_QWEN_IMAGE_TE_PATH", "")
_DEFAULT_QWEN_IMAGE_EDIT_DIT_PATH = _env_or_default("BRACKET_QWEN_IMAGE_EDIT_DIT_PATH", "")
# SD3.5 — bundle file usually contains MMDiT + T5 + CLIP
_DEFAULT_SD35_PRETRAINED = _env_or_default("BRACKET_SD35_PRETRAINED", "")
# HunyuanVideo + FramePack — share dual TE (LLaMA3 + CLIP-L)
_DEFAULT_HUNYUAN_VIDEO_DIT_PATH = _env_or_default("BRACKET_HUNYUAN_VIDEO_DIT_PATH", "")
_DEFAULT_HUNYUAN_VIDEO_VAE_PATH = _env_or_default("BRACKET_HUNYUAN_VIDEO_VAE_PATH", "")
_DEFAULT_LLAMA3_PATH = _env_or_default("BRACKET_LLAMA3_PATH", "")
_DEFAULT_FRAMEPACK_DIT_PATH = _env_or_default("BRACKET_FRAMEPACK_DIT_PATH", "")
# Wan — single TE (UMT5-XXL)
_DEFAULT_WAN_DIT_PATH = _env_or_default("BRACKET_WAN_DIT_PATH", "")
_DEFAULT_WAN_VAE_PATH = _env_or_default("BRACKET_WAN_VAE_PATH", "")
_DEFAULT_UMT5_PATH = _env_or_default("BRACKET_UMT5_PATH", "")
# LTX-Video
_DEFAULT_LTX_VIDEO_DIT_PATH = _env_or_default("BRACKET_LTX_VIDEO_DIT_PATH", "")
_DEFAULT_LTX_VIDEO_VAE_PATH = _env_or_default("BRACKET_LTX_VIDEO_VAE_PATH", "")


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


def _build_flux1_lora(**kw: Any) -> Trainer:
    from bracket.trainer.flux1_lora import Flux1LoRATrainer
    return Flux1LoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        t5xxl_path=kw["t5xxl_path"], clip_l_path=kw["clip_l_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_flux1_full(**kw: Any) -> Trainer:
    from bracket.trainer.flux1_full import Flux1FullTrainer
    return Flux1FullTrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        t5xxl_path=kw["t5xxl_path"], clip_l_path=kw["clip_l_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_flux1_kontext_lora(**kw: Any) -> Trainer:
    from bracket.trainer.flux1_kontext_lora import Flux1KontextLoRATrainer
    return Flux1KontextLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        t5xxl_path=kw["t5xxl_path"], clip_l_path=kw["clip_l_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_qwen_image_lora(**kw: Any) -> Trainer:
    from bracket.trainer.qwen_image_lora import QwenImageLoRATrainer
    return QwenImageLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_qwen_image_full(**kw: Any) -> Trainer:
    from bracket.trainer.qwen_image_full import QwenImageFullTrainer
    return QwenImageFullTrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_qwen_image_edit_lora(**kw: Any) -> Trainer:
    from bracket.trainer.qwen_image_edit_lora import QwenImageEditLoRATrainer
    return QwenImageEditLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_sd35_lora(**kw: Any) -> Trainer:
    from bracket.trainer.sd35_lora import SD35LoRATrainer
    return SD35LoRATrainer(
        sd_scripts_dir=kw["sd_scripts_dir"], venv_python=kw["venv_python"],
        pretrained_model=kw["pretrained_model"],
        model_class=kw.get("model_class", "sd35-medium"),
        vram_gb=kw.get("vram_gb"),
    )


def _build_sd35_full(**kw: Any) -> Trainer:
    from bracket.trainer.sd35_full import SD35FullTrainer
    return SD35FullTrainer(
        sd_scripts_dir=kw["sd_scripts_dir"], venv_python=kw["venv_python"],
        pretrained_model=kw["pretrained_model"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_hunyuan_video_lora(**kw: Any) -> Trainer:
    from bracket.trainer.hunyuan_video_lora import HunyuanVideoLoRATrainer
    return HunyuanVideoLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder1_path=kw["text_encoder1_path"],
        text_encoder2_path=kw["text_encoder2_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_hunyuan_video_full(**kw: Any) -> Trainer:
    from bracket.trainer.hunyuan_video_full import HunyuanVideoFullTrainer
    return HunyuanVideoFullTrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder1_path=kw["text_encoder1_path"],
        text_encoder2_path=kw["text_encoder2_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_wan_lora(**kw: Any) -> Trainer:
    from bracket.trainer.wan_lora import WanLoRATrainer
    return WanLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        wan_version=kw.get("wan_version", "2.2"),
        task=kw.get("task", "t2v-14B"),
        vram_gb=kw.get("vram_gb"),
    )


def _build_wan_full(**kw: Any) -> Trainer:
    from bracket.trainer.wan_full import WanFullTrainer
    return WanFullTrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        wan_version=kw.get("wan_version", "2.2"),
        task=kw.get("task", "t2v-14B"),
        vram_gb=kw.get("vram_gb"),
    )


def _build_ltx_video_lora(**kw: Any) -> Trainer:
    from bracket.trainer.ltx_video_lora import LTXVideoLoRATrainer
    return LTXVideoLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_framepack_lora(**kw: Any) -> Trainer:
    from bracket.trainer.framepack_lora import FramePackLoRATrainer
    return FramePackLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder1_path=kw["text_encoder1_path"],
        text_encoder2_path=kw["text_encoder2_path"],
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
    # ─────────────────────────── Flux.1 ───────────────────────────
    ModelPreset(
        id="flux1-lora",
        model_family="Flux.1",
        training_type="LoRA",
        display_name="Flux.1 (dev/schnell) · LoRA",
        trainer_factory=_build_flux1_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights (.safetensors) *",
                default=_DEFAULT_FLUX1_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_FLUX1_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE / AE weights *",
                default=_DEFAULT_FLUX1_AE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Flux.1 uses an AutoEncoder (passed as --ae). Set BRACKET_FLUX1_AE_PATH.",
            ),
            FieldSpec(
                name="t5xxl_path", label="T5-XXL text encoder *",
                default=_DEFAULT_T5XXL_PATH, required=True, kind="path",
                help="Set BRACKET_T5XXL_PATH to change the default.",
            ),
            FieldSpec(
                name="clip_l_path", label="CLIP-L text encoder *",
                default=_DEFAULT_CLIP_L_PATH, required=True, kind="path",
                help="Set BRACKET_CLIP_L_PATH to change the default.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/flux_train_network.py`. Dual TE (T5-XXL + CLIP-L). Pre-cache runs automatically.",
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="flux1-full",
        model_family="Flux.1",
        training_type="Full FT",
        display_name="Flux.1 (dev/schnell) · Full FT",
        trainer_factory=_build_flux1_full,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights (.safetensors) *",
                default=_DEFAULT_FLUX1_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_FLUX1_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE / AE weights *",
                default=_DEFAULT_FLUX1_AE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="t5xxl_path", label="T5-XXL text encoder *",
                default=_DEFAULT_T5XXL_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="clip_l_path", label="CLIP-L text encoder *",
                default=_DEFAULT_CLIP_L_PATH, required=True, kind="path",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/flux_train.py`. Full FT with Adafactor + fused backward + blocks_to_swap.",
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="flux1-kontext-lora",
        model_family="Flux.1-Kontext",
        training_type="LoRA",
        display_name="Flux.1-Kontext · LoRA",
        trainer_factory=_build_flux1_kontext_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="Kontext DiT weights *",
                default=_DEFAULT_FLUX1_KONTEXT_DIT_PATH or _DEFAULT_FLUX1_DIT_PATH,
                required=True, kind="path",
                help="Set BRACKET_FLUX1_KONTEXT_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE / AE weights *",
                default=_DEFAULT_FLUX1_AE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="t5xxl_path", label="T5-XXL text encoder *",
                default=_DEFAULT_T5XXL_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="clip_l_path", label="CLIP-L text encoder *",
                default=_DEFAULT_CLIP_L_PATH, required=True, kind="path",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/flux_kontext_train_network.py`. Trains an "
            "image-edit LoRA. Dataset TOML must declare both source (control) and "
            "target image directories."
        ),
        needs_pre_cache=True,
    ),
    # ─────────────────────────── Qwen-Image ───────────────────────────
    ModelPreset(
        id="qwen-image-lora",
        model_family="Qwen-Image",
        training_type="LoRA",
        display_name="Qwen-Image 20B · LoRA",
        trainer_factory=_build_qwen_image_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights (.safetensors) *",
                default=_DEFAULT_QWEN_IMAGE_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_QWEN_IMAGE_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_QWEN_IMAGE_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Set BRACKET_QWEN_IMAGE_VAE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Qwen2.5-VL-7B) *",
                default=_DEFAULT_QWEN_IMAGE_TE_PATH, required=True, kind="path",
                help="Qwen-Image uses Qwen2.5-VL-7B as TE (NOT plain Qwen3). Set BRACKET_QWEN_IMAGE_TE_PATH.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/qwen_image_train_network.py`. Pre-cache runs automatically.",
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="qwen-image-full",
        model_family="Qwen-Image",
        training_type="Full FT",
        display_name="Qwen-Image 20B · Full FT",
        trainer_factory=_build_qwen_image_full,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights (.safetensors) *",
                default=_DEFAULT_QWEN_IMAGE_DIT_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_QWEN_IMAGE_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Qwen2.5-VL-7B) *",
                default=_DEFAULT_QWEN_IMAGE_TE_PATH, required=True, kind="path",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/qwen_image_train.py`. Adafactor + fused + blocks_to_swap.",
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="qwen-image-edit-lora",
        model_family="Qwen-Image-Edit",
        training_type="LoRA",
        display_name="Qwen-Image-Edit · LoRA",
        trainer_factory=_build_qwen_image_edit_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="Edit DiT weights *",
                default=_DEFAULT_QWEN_IMAGE_EDIT_DIT_PATH or _DEFAULT_QWEN_IMAGE_DIT_PATH,
                required=True, kind="path",
                help="Set BRACKET_QWEN_IMAGE_EDIT_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_QWEN_IMAGE_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Qwen2.5-VL-7B) *",
                default=_DEFAULT_QWEN_IMAGE_TE_PATH, required=True, kind="path",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/qwen_image_edit_train_network.py`. Dataset TOML "
            "must declare paired source + target image directories."
        ),
        needs_pre_cache=True,
    ),
    # ─────────────────────────── SD3.5 ───────────────────────────
    ModelPreset(
        id="sd35-lora",
        model_family="SD3.5",
        training_type="LoRA",
        display_name="SD3.5 (Medium / Large) · LoRA",
        trainer_factory=_build_sd35_lora,
        fields=(
            FieldSpec(
                name="pretrained_model", label="SD3.5 base path *",
                default=_DEFAULT_SD35_PRETRAINED, required=True, kind="path",
                help="HF folder OR single .safetensors bundling MMDiT + T5 + CLIPs. Set BRACKET_SD35_PRETRAINED.",
            ),
            _SD_SCRIPTS_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `sd-scripts/sd3_train_network.py` (sd3 branch). Caches latents + TE outputs "
            "on the fly inside the trainer."
        ),
        needs_pre_cache=False,
    ),
    ModelPreset(
        id="sd35-full",
        model_family="SD3.5",
        training_type="Full FT",
        display_name="SD3.5 (Medium / Large) · Full FT",
        trainer_factory=_build_sd35_full,
        fields=(
            FieldSpec(
                name="pretrained_model", label="SD3.5 base path *",
                default=_DEFAULT_SD35_PRETRAINED, required=True, kind="path",
            ),
            _SD_SCRIPTS_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `sd-scripts/sd3_train.py`. Adafactor + fused backward.",
        needs_pre_cache=False,
    ),
    # ─────────────────────────── HunyuanVideo ───────────────────────────
    ModelPreset(
        id="hunyuan-video-lora",
        model_family="HunyuanVideo",
        training_type="LoRA",
        display_name="HunyuanVideo 13B · LoRA",
        trainer_factory=_build_hunyuan_video_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_HUNYUAN_VIDEO_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_HUNYUAN_VIDEO_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_HUNYUAN_VIDEO_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Set BRACKET_HUNYUAN_VIDEO_VAE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder1_path", label="Text encoder 1 (LLaMA3) *",
                default=_DEFAULT_LLAMA3_PATH, required=True, kind="path",
                help="Set BRACKET_LLAMA3_PATH.",
            ),
            FieldSpec(
                name="text_encoder2_path", label="Text encoder 2 (CLIP-L) *",
                default=_DEFAULT_CLIP_L_PATH, required=True, kind="path",
                help="Set BRACKET_CLIP_L_PATH.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/hv_train_network.py`. Video samples (.mp4); bracket extracts "
            "representative frames before VLM judging. ffmpeg must be on PATH."
        ),
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="hunyuan-video-full",
        model_family="HunyuanVideo",
        training_type="Full FT",
        display_name="HunyuanVideo 13B · Full FT",
        trainer_factory=_build_hunyuan_video_full,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_HUNYUAN_VIDEO_DIT_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_HUNYUAN_VIDEO_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder1_path", label="Text encoder 1 (LLaMA3) *",
                default=_DEFAULT_LLAMA3_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder2_path", label="Text encoder 2 (CLIP-L) *",
                default=_DEFAULT_CLIP_L_PATH, required=True, kind="path",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/hv_train.py`. 13B + video — heavy blocks_to_swap below 80 GB.",
        needs_pre_cache=True,
    ),
    # ─────────────────────────── Wan ───────────────────────────
    ModelPreset(
        id="wan22-lora",
        model_family="Wan 2.2",
        training_type="LoRA",
        display_name="Wan 2.2 · LoRA",
        trainer_factory=lambda **kw: _build_wan_lora(wan_version="2.2", **kw),
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_WAN_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_WAN_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_WAN_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Set BRACKET_WAN_VAE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (UMT5-XXL) *",
                default=_DEFAULT_UMT5_PATH, required=True, kind="path",
                help="Set BRACKET_UMT5_PATH.",
            ),
            FieldSpec(
                name="task", label="Task (t2v-14B / i2v-14B / t2i-14B / 1_3B)",
                default="t2v-14B", required=False, kind="string",
                help="Wan training task selector. Defaults to text-to-video 14B.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/wan_train_network.py`. UMT5-XXL TE. Video samples.",
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="wan22-full",
        model_family="Wan 2.2",
        training_type="Full FT",
        display_name="Wan 2.2 · Full FT",
        trainer_factory=lambda **kw: _build_wan_full(wan_version="2.2", **kw),
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_WAN_DIT_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_WAN_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (UMT5-XXL) *",
                default=_DEFAULT_UMT5_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="task", label="Task",
                default="t2v-14B", required=False, kind="string",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/wan_train.py`. Adafactor + fused + heavy blocks_to_swap.",
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="wan21-lora",
        model_family="Wan 2.1",
        training_type="LoRA",
        display_name="Wan 2.1 · LoRA",
        trainer_factory=lambda **kw: _build_wan_lora(wan_version="2.1", **kw),
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_WAN_DIT_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_WAN_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (UMT5-XXL) *",
                default=_DEFAULT_UMT5_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="task", label="Task",
                default="t2v-14B", required=False, kind="string",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes="Wraps `musubi-tuner/wan_train_network.py` against Wan 2.1 weights.",
        needs_pre_cache=True,
    ),
    # ─────────────────────────── LTX-Video ───────────────────────────
    ModelPreset(
        id="ltx-video-lora",
        model_family="LTX-Video",
        training_type="LoRA",
        display_name="LTX-Video · LoRA",
        trainer_factory=_build_ltx_video_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_LTX_VIDEO_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_LTX_VIDEO_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_LTX_VIDEO_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Set BRACKET_LTX_VIDEO_VAE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (T5-XXL) *",
                default=_DEFAULT_T5XXL_PATH, required=True, kind="path",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/ltxv_train_network.py`. Smallest of the supported "
            "video DiTs (~2B) — much faster per-step than Wan / Hunyuan."
        ),
        needs_pre_cache=True,
    ),
    # ─────────────────────────── FramePack ───────────────────────────
    ModelPreset(
        id="framepack-lora",
        model_family="FramePack",
        training_type="LoRA",
        display_name="FramePack · LoRA",
        trainer_factory=_build_framepack_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_FRAMEPACK_DIT_PATH or _DEFAULT_HUNYUAN_VIDEO_DIT_PATH,
                required=True, kind="path",
                help="Set BRACKET_FRAMEPACK_DIT_PATH.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_HUNYUAN_VIDEO_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder1_path", label="Text encoder 1 (LLaMA3) *",
                default=_DEFAULT_LLAMA3_PATH, required=True, kind="path",
            ),
            FieldSpec(
                name="text_encoder2_path", label="Text encoder 2 (CLIP-L) *",
                default=_DEFAULT_CLIP_L_PATH, required=True, kind="path",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/fpack_train_network.py`. Anchor-frame-conditioned "
            "video LoRA on the HunyuanVideo backbone."
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
