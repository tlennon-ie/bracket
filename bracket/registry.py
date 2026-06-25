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
#   2. The repo-vendored trainers under <repo>/vendor/ (git submodules,
#      pinned to specific commits — see .gitmodules and the bump workflow
#      in docs/UPDATING_TRAINERS.md).
#   3. The legacy ~/.cache/bracket/trainers/ location for users who
#      installed before the submodule layout existed.
#   4. An empty string — the user fills it in via the Setup tab.
#
# Nothing in this module assumes a specific developer's filesystem. Public
# users land on empty defaults if they haven't run the installer; pre-existing
# users can keep their setup by exporting the env vars.

_REPO_ROOT = Path(__file__).resolve().parent.parent
_REPO_VENDOR = _REPO_ROOT / "vendor"
_LEGACY_TRAINERS_ROOT = Path(
    os.environ.get("BRACKET_TRAINERS_ROOT", str(Path.home() / ".cache" / "bracket" / "trainers"))
)


def _env_or_default(env_var: str, default: str) -> str:
    """Return the env-var value if set, else the default. Empty string if neither."""
    return os.environ.get(env_var, default)


def _first_existing(*candidates: Path) -> str:
    """Return the first path that exists as a directory or file, else empty."""
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def _venv_python_default() -> str:
    """Resolve the trainer venv python — env var or installed location.

    Looks for the in-repo ``vendor/venv`` first (current install layout),
    then falls back to the legacy user-cache location.
    """
    if (env := os.environ.get("BRACKET_VENV_PYTHON")):
        return env
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return _first_existing(
        _REPO_VENDOR / "venv" / suffix,
        _LEGACY_TRAINERS_ROOT / "venv" / suffix,
    )


def _musubi_dir_default() -> str:
    if (env := os.environ.get("BRACKET_MUSUBI_DIR")):
        return env
    return _first_existing(
        _REPO_VENDOR / "musubi-tuner",
        _LEGACY_TRAINERS_ROOT / "musubi-tuner",
    )


def _sd_scripts_default() -> str:
    if (env := os.environ.get("BRACKET_SD_SCRIPTS_DIR")):
        return env
    # New (submodule) layout puts sd-scripts as a sibling of musubi-tuner;
    # legacy install nested it under musubi-tuner/sd-scripts.
    return _first_existing(
        _REPO_VENDOR / "sd-scripts",
        _LEGACY_TRAINERS_ROOT / "musubi-tuner" / "sd-scripts",
    )


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
# LTX-2 (Lightricks' native ltx-trainer — YAML-driven, Gemma text encoder)
_DEFAULT_LTX2_MODEL_PATH = _env_or_default("BRACKET_LTX2_MODEL_PATH", "")
_DEFAULT_LTX2_TEXT_ENCODER_PATH = _env_or_default("BRACKET_LTX2_TEXT_ENCODER_PATH", "")
# HiDream-O1 — single-file checkpoint (--dit only); tokenizer/TE auto-load from HF
_DEFAULT_HIDREAM_DIT_PATH = _env_or_default("BRACKET_HIDREAM_DIT_PATH", "")
# HunyuanVideo 1.5 — Qwen2.5-VL TE + ByT5 glyph encoder (optional SigLIP image encoder for i2v)
_DEFAULT_HUNYUAN_VIDEO_15_DIT_PATH = _env_or_default("BRACKET_HUNYUAN_VIDEO_15_DIT_PATH", "")
_DEFAULT_HUNYUAN_VIDEO_15_VAE_PATH = _env_or_default("BRACKET_HUNYUAN_VIDEO_15_VAE_PATH", "")
_DEFAULT_HUNYUAN_VIDEO_15_QWEN_TE_PATH = _env_or_default("BRACKET_HUNYUAN_VIDEO_15_QWEN_TE_PATH", "")
_DEFAULT_HUNYUAN_VIDEO_15_BYT5_PATH = _env_or_default("BRACKET_HUNYUAN_VIDEO_15_BYT5_PATH", "")
# FLUX.2-dev — full FLUX.2 (distinct from the Klein 9B distill); TE reuses Mistral-3
_DEFAULT_FLUX2_DEV_DIT_PATH = _env_or_default("BRACKET_FLUX2_DEV_DIT_PATH", "")
_DEFAULT_FLUX2_DEV_VAE_PATH = _env_or_default("BRACKET_FLUX2_DEV_VAE_PATH", "")
# Kandinsky 5 — dual TE (Qwen2.5-VL + CLIP-L)
_DEFAULT_KANDINSKY5_DIT_PATH = _env_or_default("BRACKET_KANDINSKY5_DIT_PATH", "")
_DEFAULT_KANDINSKY5_VAE_PATH = _env_or_default("BRACKET_KANDINSKY5_VAE_PATH", "")
_DEFAULT_KANDINSKY5_QWEN_TE_PATH = _env_or_default("BRACKET_KANDINSKY5_QWEN_TE_PATH", "")
_DEFAULT_KANDINSKY5_CLIP_TE_PATH = _env_or_default("BRACKET_KANDINSKY5_CLIP_TE_PATH", "")


def _ltx2_trainer_dir_default() -> str:
    """Resolve the ltx-trainer package dir — env var or repo-vendored layout."""
    if (env := os.environ.get("BRACKET_LTX2_TRAINER_DIR")):
        return env
    return _first_existing(_REPO_VENDOR / "ltx2" / "packages" / "ltx-trainer")


_DEFAULT_LTX2_TRAINER_DIR = _ltx2_trainer_dir_default()


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
        sd_scripts_dir=kw["sd_scripts_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        t5xxl_path=kw["t5xxl_path"], clip_l_path=kw["clip_l_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_flux1_full(**kw: Any) -> Trainer:
    from bracket.trainer.flux1_full import Flux1FullTrainer
    return Flux1FullTrainer(
        sd_scripts_dir=kw["sd_scripts_dir"], venv_python=kw["venv_python"],
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
        model_version=kw.get("model_version", "edit-2509"),
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


def _build_ltx2_t2v_lora(**kw: Any) -> Trainer:
    from bracket.trainer.ltx2_lora import LTX2LoRATrainer
    return LTX2LoRATrainer(
        trainer_dir=kw["trainer_dir"],
        model_path=kw["model_path"],
        text_encoder_path=kw["text_encoder_path"],
        mode="t2v",
        vram_gb=kw.get("vram_gb"),
    )


def _build_ltx2_i2v_lora(**kw: Any) -> Trainer:
    from bracket.trainer.ltx2_lora import LTX2LoRATrainer
    return LTX2LoRATrainer(
        trainer_dir=kw["trainer_dir"],
        model_path=kw["model_path"],
        text_encoder_path=kw["text_encoder_path"],
        mode="i2v",
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


def _build_hidream_lora(**kw: Any) -> Trainer:
    from bracket.trainer.hidream_lora import HiDreamLoRATrainer
    return HiDreamLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_hunyuan_video_15_lora(**kw: Any) -> Trainer:
    from bracket.trainer.hunyuan_video_15_lora import HunyuanVideo15LoRATrainer
    return HunyuanVideo15LoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        byt5_path=kw["byt5_path"],
        task="t2v",
        vram_gb=kw.get("vram_gb"),
    )


def _build_flux2_dev_lora(**kw: Any) -> Trainer:
    from bracket.trainer.flux2_dev_lora import Flux2DevLoRATrainer
    return Flux2DevLoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_path=kw["text_encoder_path"],
        vram_gb=kw.get("vram_gb"),
    )


def _build_kandinsky5_lora(**kw: Any) -> Trainer:
    from bracket.trainer.kandinsky5_lora import Kandinsky5LoRATrainer
    return Kandinsky5LoRATrainer(
        musubi_dir=kw["musubi_dir"], venv_python=kw["venv_python"],
        dit_path=kw["dit_path"], vae_path=kw["vae_path"],
        text_encoder_qwen_path=kw["text_encoder_qwen_path"],
        text_encoder_clip_path=kw["text_encoder_clip_path"],
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
    help="Path to the sd-scripts (Kohya) checkout. Vendored at vendor/sd-scripts; set BRACKET_SD_SCRIPTS_DIR to override.",
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

_LTX2_TRAINER_DIR_FIELD = FieldSpec(
    name="trainer_dir", label="ltx-trainer directory *",
    default=_DEFAULT_LTX2_TRAINER_DIR, required=True, kind="dir",
    help=(
        "Path to Lightricks' native ltx-trainer package "
        "(<ltx2>/packages/ltx-trainer). Set BRACKET_LTX2_TRAINER_DIR to change "
        "the default."
    ),
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
    ModelPreset(
        id="flux2-dev-lora",
        model_family="FLUX.2",
        training_type="LoRA",
        display_name="FLUX.2-dev · LoRA",
        trainer_factory=_build_flux2_dev_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights (.safetensors) *",
                default=_DEFAULT_FLUX2_DEV_DIT_PATH,
                required=True, kind="path",
                help="Set BRACKET_FLUX2_DEV_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights (.safetensors) *",
                default=_DEFAULT_FLUX2_DEV_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Set BRACKET_FLUX2_DEV_VAE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Mistral-3) *",
                default=_DEFAULT_MISTRAL3_TE_PATH,
                required=True, kind="path",
                help="FLUX.2 uses Mistral-3 as the text encoder. Set BRACKET_MISTRAL3_TE_PATH.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/flux_2_train_network.py` with `--model_version dev`. "
            "Full FLUX.2 (~32B) — distinct from the distilled Flux-2-Klein 9B preset "
            "above. Mistral-3 text encoder, fp8 base + scaled pinned on. Pre-cache "
            "runs automatically."
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
            _SD_SCRIPTS_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `sd-scripts/flux_train_network.py`. Dual TE (T5-XXL + CLIP-L) "
            "+ AutoEncoder (`--ae`). Caches latents + TE outputs inline inside the "
            "trainer — no manual pre-cache step."
        ),
        needs_pre_cache=False,
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
            _SD_SCRIPTS_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `sd-scripts/flux_train.py`. Full FT with Adafactor + fused "
            "backward + full_bf16 + blocks_to_swap. Caches latents inline — no "
            "manual pre-cache step."
        ),
        needs_pre_cache=False,
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
        notes=(
            "Wraps `musubi-tuner/qwen_image_train_network.py` "
            "(`--model_version original`, the default). Pre-cache runs "
            "automatically. The newer **Qwen-Image-2512** base checkpoint is a "
            "drop-in: point the DiT path at the 2512 weights — same architecture, "
            "no flag changes."
        ),
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
            FieldSpec(
                name="model_version", label="Edit version (edit / edit-2509 / edit-2511)",
                default="edit-2509", required=False, kind="string",
                help=(
                    "Selects the Qwen-Image-Edit architecture via --model_version. "
                    "`edit` = original (single control image); `edit-2509` / "
                    "`edit-2511` = newer multi-reference checkpoints (up to 3 "
                    "control images). Match this to your DiT weights."
                ),
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/qwen_image_train_network.py` with "
            "`--model_version edit|edit-2509|edit-2511` (upstream unified the "
            "edit path onto the plain Qwen-Image scripts). Point the Edit DiT "
            "path at any Edit checkpoint — **Edit-2509 and Edit-2511** are "
            "supported by setting the Edit-version selector; 2509/2511 add "
            "multi-reference editing (up to 3 control images). Dataset TOML must "
            "declare paired source + target image directories."
        ),
        needs_pre_cache=True,
    ),
    # ─────────────────────────── HiDream-O1 ───────────────────────────
    ModelPreset(
        id="hidream-lora",
        model_family="HiDream",
        training_type="LoRA",
        display_name="HiDream-O1 · LoRA",
        trainer_factory=_build_hidream_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights (.safetensors) *",
                default=_DEFAULT_HIDREAM_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_HIDREAM_DIT_PATH to change the default.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/hidream_o1_train_network.py`. Single-file "
            "checkpoint — only `--dit` is needed; the tokenizer / text encoder "
            "auto-load from the official HiDream-ai HF repos. No VAE or TE path. "
            "Pre-cache runs automatically."
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
    # ─────────────────────────── HunyuanVideo 1.5 ───────────────────────────
    ModelPreset(
        id="hunyuan-video-15-lora",
        model_family="HunyuanVideo 1.5",
        training_type="LoRA",
        display_name="HunyuanVideo 1.5 · LoRA",
        trainer_factory=_build_hunyuan_video_15_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_HUNYUAN_VIDEO_15_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_HUNYUAN_VIDEO_15_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_HUNYUAN_VIDEO_15_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Set BRACKET_HUNYUAN_VIDEO_15_VAE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_path", label="Text encoder (Qwen2.5-VL) *",
                default=_DEFAULT_HUNYUAN_VIDEO_15_QWEN_TE_PATH, required=True, kind="path",
                help="HunyuanVideo 1.5 uses Qwen2.5-VL as TE. Set BRACKET_HUNYUAN_VIDEO_15_QWEN_TE_PATH.",
            ),
            FieldSpec(
                name="byt5_path", label="ByT5 glyph encoder *",
                default=_DEFAULT_HUNYUAN_VIDEO_15_BYT5_PATH, required=True, kind="path",
                help="Set BRACKET_HUNYUAN_VIDEO_15_BYT5_PATH to change the default.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/hv_1_5_train_network.py` (`--task t2v`). Next-gen "
            "Tencent video DiT — Qwen2.5-VL TE + ByT5 glyph encoder (distinct from "
            "the dual-TE HunyuanVideo 1.0). Trains at 720p / 121 frames. Video "
            "samples (.mp4); bracket extracts frames before VLM judging. Pre-cache "
            "runs automatically."
        ),
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
        notes=(
            "Wraps `musubi-tuner/wan_train_network.py`. UMT5-XXL TE. Video "
            "samples. Wan 2.2 support in musubi-tuner is the **14B MoE** "
            "(`--task t2v-A14B` / `i2v-A14B`, dual high/low-noise DiT). The "
            "smaller **TI2V-5B** single model is NOT yet trainable: upstream "
            "lists `ti2v-5B` only in the size table — it has no entry in "
            "`WAN_CONFIGS`, so `wan_train_network.py --task ti2v-5B` is rejected. "
            "Use the A14B tasks until upstream wires up the 5B config."
        ),
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
    # ─────────────────────────── Kandinsky 5 ───────────────────────────
    ModelPreset(
        id="kandinsky5-lora",
        model_family="Kandinsky 5",
        training_type="LoRA",
        display_name="Kandinsky 5 · LoRA",
        trainer_factory=_build_kandinsky5_lora,
        fields=(
            FieldSpec(
                name="dit_path", label="DiT weights *",
                default=_DEFAULT_KANDINSKY5_DIT_PATH, required=True, kind="path",
                help="Set BRACKET_KANDINSKY5_DIT_PATH to change the default.",
            ),
            FieldSpec(
                name="vae_path", label="VAE weights *",
                default=_DEFAULT_KANDINSKY5_VAE_PATH or _DEFAULT_VAE_PATH,
                required=True, kind="path",
                help="Set BRACKET_KANDINSKY5_VAE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_qwen_path", label="Text encoder (Qwen2.5-VL) *",
                default=_DEFAULT_KANDINSKY5_QWEN_TE_PATH, required=True, kind="path",
                help="Set BRACKET_KANDINSKY5_QWEN_TE_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_clip_path", label="Text encoder (CLIP-L) *",
                default=_DEFAULT_KANDINSKY5_CLIP_TE_PATH, required=True, kind="path",
                help="Set BRACKET_KANDINSKY5_CLIP_TE_PATH to change the default.",
            ),
            _MUSUBI_DIR_FIELD,
            _VENV_PYTHON_FIELD,
        ),
        notes=(
            "Wraps `musubi-tuner/kandinsky5_train_network.py`. Sber's flow-matching "
            "model with dual text encoders (Qwen2.5-VL + CLIP-L). The `--task` "
            "selector is pinned to the default config. Pre-cache runs automatically."
        ),
        needs_pre_cache=True,
    ),
    # ─────────────────────────── LTX-2 (native ltx-trainer) ───────────────────────────
    ModelPreset(
        id="ltx2-t2v-lora",
        model_family="LTX-2",
        training_type="LoRA",
        display_name="LTX-2 · T2V LoRA",
        trainer_factory=_build_ltx2_t2v_lora,
        fields=(
            FieldSpec(
                name="model_path", label="LTX-2 model path *",
                default=_DEFAULT_LTX2_MODEL_PATH, required=True, kind="path",
                help="LTX-2 model weights/checkpoint. Set BRACKET_LTX2_MODEL_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_path", label="Gemma text encoder *",
                default=_DEFAULT_LTX2_TEXT_ENCODER_PATH, required=True, kind="path",
                help="LTX-2 uses Gemma as its text encoder. Set BRACKET_LTX2_TEXT_ENCODER_PATH.",
            ),
            _LTX2_TRAINER_DIR_FIELD,
        ),
        notes=(
            "Drives Lightricks' **native** `ltx-trainer` (`scripts/train.py`) — "
            "YAML-config-driven, Gemma text encoder, joint audio-video. "
            "Preprocessing (`scripts/process_dataset.py`) runs once per session "
            "into a deterministic cache reused by every candidate. The newer "
            "**LTX-2.3** checkpoint is a drop-in: point the LTX-2 model path at "
            "the 2.3 weights — no flag changes."
        ),
        needs_pre_cache=True,
    ),
    ModelPreset(
        id="ltx2-i2v-lora",
        model_family="LTX-2",
        training_type="LoRA",
        display_name="LTX-2 · I2V LoRA",
        trainer_factory=_build_ltx2_i2v_lora,
        fields=(
            FieldSpec(
                name="model_path", label="LTX-2 model path *",
                default=_DEFAULT_LTX2_MODEL_PATH, required=True, kind="path",
                help="LTX-2 model weights/checkpoint. Set BRACKET_LTX2_MODEL_PATH to change the default.",
            ),
            FieldSpec(
                name="text_encoder_path", label="Gemma text encoder *",
                default=_DEFAULT_LTX2_TEXT_ENCODER_PATH, required=True, kind="path",
                help="LTX-2 uses Gemma as its text encoder. Set BRACKET_LTX2_TEXT_ENCODER_PATH.",
            ),
            _LTX2_TRAINER_DIR_FIELD,
        ),
        notes=(
            "Image-to-video variant of the native LTX-2 `ltx-trainer`. Adds a "
            "`first_frame` condition to the flexible training strategy. "
            "YAML-config-driven, Gemma text encoder, joint audio-video. "
            "Preprocessing runs once per session into a deterministic cache. The "
            "newer **LTX-2.3** checkpoint is a drop-in: point the LTX-2 model "
            "path at the 2.3 weights — no flag changes."
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
