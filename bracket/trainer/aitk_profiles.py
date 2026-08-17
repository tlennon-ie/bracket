"""ai-toolkit model profiles — the per-architecture config deltas.

ostris' ``ai-toolkit`` drives every model through one ``sd_trainer`` job; what
differs between models is not the *shape* of the YAML but a handful of
per-architecture values scattered across its ``model`` / ``train`` /
``datasets`` / ``network`` / ``sample`` blocks. :mod:`bracket.trainer.aitk_lora`
owns the shape; this module owns the values.

Each :class:`AiToolkitProfile` is transcribed from that architecture's entry in
ai-toolkit's own ``ui/src/app/jobs/new/options.tsx`` (the authoritative
per-arch default table its web UI writes into a job config) — not invented
here. When ai-toolkit bumps, re-diff that file, not the model classes.

Three media kinds are covered:

``image``
    Chroma, Lumina2, OmniGen2, Flex.1, Flex.2. Single frames; the VLM judge
    scores samples directly.

``video``
    LTX-2.5, MiniMax-H3, MiniMax-H3 Ref2VA. ``datasets[].num_frames > 1`` is
    what switches ai-toolkit's dataloader from images to video clips — leaving
    it at ai-toolkit's own default of ``1`` silently trains a video model on
    stills, so every video profile sets it explicitly. Samples are animated
    files; Bracket's scorer extracts frames before judging.

``audio``
    ACE-Step 1.5 and 1.5 XL (music). Samples are audio — the VLM judge cannot
    score them, so these runs are ranked on the loss curve alone. See the
    preset notes in :mod:`bracket.registry`.

Adding an arch = add a profile here + a ``ModelPreset`` in
:mod:`bracket.registry`. No change to the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# The three media kinds. ``video`` and ``audio`` change which blocks the
# adapter emits; ``image`` is the historical (and default) behaviour.
MEDIA_IMAGE = "image"
MEDIA_VIDEO = "video"
MEDIA_AUDIO = "audio"
MEDIA_KINDS = (MEDIA_IMAGE, MEDIA_VIDEO, MEDIA_AUDIO)


@dataclass(frozen=True)
class AiToolkitProfile:
    """One ai-toolkit architecture's deltas against the shared job config.

    Attributes:
        model_id: Short label — names the trainer + search space
            (``aitk-lora-<model_id>``) and keys :data:`PROFILES`.
        default_model: Default ``model.name_or_path`` (HF repo id or a
            repo-relative weight file). Shown as the preset's field default.
        media_kind: One of :data:`MEDIA_KINDS`.
        model_extra: Merged into the ``model:`` block. Carries the arch
            SELECTOR verbatim — most models use ``arch: <name>``, but Lumina2
            uses ``is_lumina2: true`` and Flex.1 uses ``is_flux: true``, so
            this is never derived from ``model_id``.
        train_extra: Merged into the ``train:`` block.
        dataset_extra: Merged into the single ``datasets[0]`` entry.
        network_extra: Merged into the ``network:`` block (rank/alpha stay
            searchable and are set by the adapter).
        sample_extra: Merged into the ``sample:`` block.
    """

    model_id: str
    default_model: str
    media_kind: str = MEDIA_IMAGE
    model_extra: Mapping[str, Any] = field(default_factory=dict)
    train_extra: Mapping[str, Any] = field(default_factory=dict)
    dataset_extra: Mapping[str, Any] = field(default_factory=dict)
    network_extra: Mapping[str, Any] = field(default_factory=dict)
    sample_extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.media_kind not in MEDIA_KINDS:
            raise ValueError(
                f"{self.model_id}: media_kind must be one of {MEDIA_KINDS}, "
                f"got {self.media_kind!r}"
            )


# ─────────────────────────── image (v0.1 set) ───────────────────────────

_IMAGE_PROFILES: tuple[AiToolkitProfile, ...] = (
    AiToolkitProfile(
        model_id="chroma",
        default_model="lodestones/Chroma",
        model_extra={"arch": "chroma", "quantize": True},
    ),
    AiToolkitProfile(
        model_id="lumina2",
        default_model="Alpha-VLLM/Lumina-Image-2.0",
        # No ``lumina2`` entry exists in ai-toolkit's arch registry — the
        # selector is this boolean. ``arch: lumina2`` would be rejected.
        model_extra={"is_lumina2": True, "quantize_te": True},
    ),
    AiToolkitProfile(
        model_id="omnigen2",
        default_model="OmniGen2/OmniGen2",
        model_extra={"arch": "omnigen2", "quantize_te": True},
    ),
    AiToolkitProfile(
        model_id="flex1",
        default_model="ostris/Flex.1-alpha",
        model_extra={"is_flux": True, "quantize": True},
    ),
    AiToolkitProfile(
        model_id="flex2",
        default_model="ostris/Flex.2-preview",
        model_extra={"arch": "flex2", "quantize": True, "quantize_te": True},
    ),
)


# ─────────────────────────── video ───────────────────────────

# MiniMax-H3's VAE quantises clip length onto a 17n+5 grid; 39 = 17*2+5 is
# ai-toolkit's own dataset default for the arch.
_MINIMAX_H3_DATASET_FRAMES = 39
# LTX-2's VAE uses an 8n+1 grid. ai-toolkit ships no dataset-level default for
# LTX-2.5 (so its dataloader falls back to 1 = stills); 49 = 8*6+1 is Bracket's
# pick — ~2 s at 24 fps, the shortest clip length that still trains motion
# without the VRAM cost of the 121-frame sample length.
_LTX25_DATASET_FRAMES = 49

# MiniMax-H3 and its Ref2VA sibling are guidance-distilled: training on them
# directly degrades the distillation. ai-toolkit's default remedy is BOTH
# contrastive-guidance loss and the published training adapter, which is what
# these dicts encode.
_MINIMAX_H3_TRAIN: dict[str, Any] = {
    "cache_text_embeddings": True,
    "do_guidance_loss": True,
    "guidance_loss_target": 3.5,
    "audio_loss_multiplier": 1.0,
    "timestep_type": "shift",
}
# The adaln projection is excluded from the LoRA — training it fights the
# distillation adapter.
_MINIMAX_H3_NETWORK: dict[str, Any] = {
    "network_kwargs": {"ignore_if_contains": ["adaln_proj"]},
}
_MINIMAX_H3_SAMPLE: dict[str, Any] = {
    "num_frames": 107,
    "fps": 24,
    "width": 768,
    "height": 768,
    # Distilled model — sampling guidance is 1, not the image-model default.
    "guidance_scale": 1,
    "sample_steps": 28,
}
# The Comfy-Org weights ship pre-quantised (int8 ConvRot DiT, nvfp4 TE). These
# qtypes MATCH the checkpoints, so the load is a straight read; naming a
# different qtype would re-quantise layer by layer on every run.
_MINIMAX_H3_MODEL: dict[str, Any] = {
    "quantize": True,
    "qtype": "convrot8",
    "quantize_te": True,
    "qtype_te": "nvfp4",
    "low_vram": True,
}

_VIDEO_PROFILES: tuple[AiToolkitProfile, ...] = (
    AiToolkitProfile(
        model_id="ltx25",
        default_model="Lightricks/LTX-2.5",
        media_kind=MEDIA_VIDEO,
        model_extra={
            "arch": "ltx2.5",
            # Comfy-style split files; the int8 ConvRot dev transformer is the
            # default the arch resolves to.
            "quantize": True,
            "qtype": "convrot8",
            "quantize_te": True,
            "qtype_te": "convrot8",
            "low_vram": True,
        },
        train_extra={"timestep_type": "weighted", "audio_loss_multiplier": 1.0},
        dataset_extra={
            "num_frames": _LTX25_DATASET_FRAMES,
            "fps": 24,
            "do_audio": True,
            "do_i2v": False,
            "auto_frame_count": False,
        },
        sample_extra={"num_frames": 121, "fps": 24, "width": 768, "height": 768},
    ),
    AiToolkitProfile(
        model_id="minimax_h3",
        default_model="Comfy-Org/MiniMax-H3",
        media_kind=MEDIA_VIDEO,
        model_extra={
            "arch": "minimax_h3",
            **_MINIMAX_H3_MODEL,
            "assistant_lora_path": (
                "ostris/minimax_h3_training_adapter/"
                "minimax_h3_training_adapter_v1.safetensors"
            ),
        },
        train_extra=dict(_MINIMAX_H3_TRAIN),
        dataset_extra={
            "num_frames": _MINIMAX_H3_DATASET_FRAMES,
            "fps": 24,
            "do_audio": True,
            "do_i2v": False,
            "auto_frame_count": True,
        },
        network_extra=dict(_MINIMAX_H3_NETWORK),
        sample_extra=dict(_MINIMAX_H3_SAMPLE),
    ),
    AiToolkitProfile(
        model_id="minimax_h3_ref2va",
        default_model="Comfy-Org/MiniMax-H3",
        media_kind=MEDIA_VIDEO,
        model_extra={
            "arch": "minimax_h3_ref2va",
            **_MINIMAX_H3_MODEL,
            "assistant_lora_path": (
                "ostris/minimax_h3_training_adapter/"
                "minimax_h3_ref2va_training_adapter_v1.safetensors"
            ),
        },
        train_extra=dict(_MINIMAX_H3_TRAIN),
        # Ref2VA conditions on reference images/videos rather than a first
        # frame, so it declares no ``do_i2v``.
        dataset_extra={
            "num_frames": _MINIMAX_H3_DATASET_FRAMES,
            "fps": 24,
            "do_audio": True,
            "auto_frame_count": True,
        },
        network_extra=dict(_MINIMAX_H3_NETWORK),
        sample_extra=dict(_MINIMAX_H3_SAMPLE),
    ),
)


# ─────────────────────────── audio (music) ───────────────────────────

_ACE_STEP_MODEL: dict[str, Any] = {
    "quantize": True,
    "qtype": "qfloat8",
    "quantize_te": True,
    "low_vram": True,
}
_ACE_STEP_TRAIN: dict[str, Any] = {
    # ACE-Step conditions on the caption at every step, so the text encoder
    # stays resident.
    "unload_text_encoder": False,
    "timestep_type": "linear",
}

_AUDIO_PROFILES: tuple[AiToolkitProfile, ...] = (
    AiToolkitProfile(
        model_id="ace_step_15",
        default_model=(
            "ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_base_aio.safetensors"
        ),
        media_kind=MEDIA_AUDIO,
        model_extra={"arch": "ace_step_15", **_ACE_STEP_MODEL},
        train_extra=dict(_ACE_STEP_TRAIN),
    ),
    AiToolkitProfile(
        model_id="ace_step_15_xl",
        default_model=(
            "ostris/ace_step_1.5_ComfyUI_files/ace_step_1.5_xl_base_aio.safetensors"
        ),
        media_kind=MEDIA_AUDIO,
        model_extra={"arch": "ace_step_15_xl", **_ACE_STEP_MODEL},
        train_extra=dict(_ACE_STEP_TRAIN),
    ),
)


PROFILES: Mapping[str, AiToolkitProfile] = {
    p.model_id: p
    for p in (*_IMAGE_PROFILES, *_VIDEO_PROFILES, *_AUDIO_PROFILES)
}


def get_profile(model_id: str) -> AiToolkitProfile:
    """Return the profile for ``model_id``.

    Raises:
        KeyError: with the known ids listed, so a typo in a preset fails at
            import time with an actionable message rather than producing a
            YAML ai-toolkit rejects at launch.
    """
    try:
        return PROFILES[model_id]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(
            f"unknown ai-toolkit profile {model_id!r}; known ids: {known}"
        ) from None
