"""GPU hardware detection — used by trainer adapters to scale search spaces
to the user's actual VRAM rather than musubi-tuner's worst-case defaults.

The baseline this module replaces was "always batch_size=1, always
gradient_checkpointing on" — fine for an 8GB card, but on a 32GB 5090 it
leaves ~20GB unused and trains slower than necessary. With detection we can
crank batch_size up to fill the card without OOMing, AND let the orchestrator
include batch_size as a real search-space knob instead of a pin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    name: str
    total_vram_gb: float
    device_index: int


def detect_gpu(device_index: int = 0) -> Optional[GPUInfo]:
    """Returns GPU info for the given device, or None if no CUDA / detection fails."""
    try:
        import torch  # imported lazily so cpu-only test runs don't pay the import cost
        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(device_index)
        return GPUInfo(
            name=props.name,
            total_vram_gb=props.total_memory / (1024 ** 3),
            device_index=device_index,
        )
    except Exception:
        logger.exception("GPU detection failed")
        return None


def vram_tier(vram_gb: float) -> str:
    """Bucket a VRAM size into a coarse tier name. Used by trainer adapters
    to make decisions like "what batch_size range is safe at this VRAM"."""
    if vram_gb >= 70:
        return "xl"        # A100 80GB, H100 80GB
    if vram_gb >= 40:
        return "large"     # A6000 48GB, A100 40GB, RTX 6000 Ada 48GB
    if vram_gb >= 28:
        return "high"      # RTX 5090 32GB, A5000 24GB, 4090 24GB (close enough)
    if vram_gb >= 20:
        return "med"       # RTX 3090 24GB, RTX 4090 24GB at edge, A4500
    if vram_gb >= 12:
        return "low"       # RTX 4070 Ti 12GB, 3060 12GB, A4000
    return "tiny"          # 8GB cards


# Per-tier batch-size choices for SDXL LoRA at 1024x1024 with bf16 +
# gradient_checkpointing on. Bounded by safe VRAM headroom (~20% reserve)
# so other GPU-using processes don't OOM the trainer.
SDXL_LORA_BATCH_CHOICES_BY_TIER: dict[str, tuple[int, ...]] = {
    "xl":    (4, 8, 16, 32),
    "large": (2, 4, 8, 16),
    "high":  (1, 2, 4, 8),
    "med":   (1, 2, 4),
    "low":   (1, 2),
    "tiny":  (1,),
}


# Sensible default ("baseline") batch size per tier — what a competent human
# would type by hand, leaving margin for safety.
SDXL_LORA_DEFAULT_BATCH_BY_TIER: dict[str, int] = {
    "xl":    8,
    "large": 4,
    "high":  4,
    "med":   2,
    "low":   2,
    "tiny":  1,
}


# DataLoader worker count by VRAM tier (sd-scripts family).
# Higher = better feed rate but more RAM. 2 was the universal default; on a
# 5090 (32GB high tier) you want 6+ to avoid stalling the GPU on disk I/O.
DATALOADER_WORKERS_BY_TIER: dict[str, int] = {
    "xl":    8,
    "large": 6,
    "high":  6,
    "med":   4,
    "low":   2,
    "tiny":  2,
}


# DataLoader worker count for musubi-tuner trainers — much lower than sd-scripts.
# Musubi's docs explicitly recommend `--max_data_loader_n_workers 2` and the
# trainer behaviour with multiple [[datasets]] entries amplifies worker count
# (one DataLoader per dataset). With persistent_data_loader_workers + 5
# datasets, num_workers=6 spawns 30 processes which thrash on most boxes.
MUSUBI_DATALOADER_WORKERS_BY_TIER: dict[str, int] = {
    "xl":    2,
    "large": 2,
    "high":  2,
    "med":   2,
    "low":   2,
    "tiny":  2,
}


# Approximate model parameter count in billions, by model class. Used to
# decide whether gradient checkpointing can safely be off at a given VRAM
# tier. Activations alone for a backward-pass without checkpointing scale
# with model size and batch size; on a 5090 (32GB), a 9B-class model at
# batch=4 fp16/bf16 will OOM without ckpt and fall back to RAM paging.
MODEL_PARAMS_B: dict[str, float] = {
    "sdxl":          2.5,   # 2.6B UNet
    "sd35-medium":   2.5,   # SD3.5 Medium (~2.5B MMDiT)
    "sd35-large":    8.0,   # SD3.5 Large (~8B MMDiT)
    "zimage":        9.0,   # 9B DiT (Tongyi-MAI/Z-Image base + Turbo)
    "flux1":        12.0,   # Flux.1 dev/schnell (~12B DiT)
    "flux1-kontext":12.0,   # Flux.1-Kontext (same backbone, edit conditioned)
    "flux2-klein":   9.0,   # Flux 2 Klein 9B (fp8 helps weights, not activations)
    "flux2-full":   32.0,
    "qwen-image":   20.0,   # Qwen-Image 20B MMDiT
    "qwen-image-edit": 20.0,
    "hunyuan-video": 13.0,  # HunyuanVideo 13B DiT
    "wan21":        14.0,   # Wan 2.1 14B (1.3B variant exists too)
    "wan22":        14.0,
    "ltx-video":     2.0,   # LTX-Video small DiT (~2B)
    "framepack":    13.0,   # FramePack uses HunyuanVideo backbone
}


# Per-tier batch-size choices for video models. Video activations are dominated
# by frame count not pixel count; bs=1 is the realistic default everywhere
# below 80 GB. These tables exist to make video adapters consistent.
VIDEO_LORA_BATCH_CHOICES_BY_TIER: dict[str, tuple[int, ...]] = {
    "xl":    (1, 2),
    "large": (1, 2),
    "high":  (1,),
    "med":   (1,),
    "low":   (1,),
    "tiny":  (1,),
}
VIDEO_LORA_DEFAULT_BATCH_BY_TIER: dict[str, int] = {
    "xl": 1, "large": 1, "high": 1, "med": 1, "low": 1, "tiny": 1,
}


# `--blocks_to_swap` lets musubi swap N transformer blocks to CPU RAM. The
# higher the number, the less VRAM but the slower training. These are sane
# starting values for big DiTs (HunyuanVideo / Wan / Qwen-Image) per VRAM tier.
BLOCKS_TO_SWAP_BY_TIER: dict[str, int] = {
    "xl":    0,
    "large": 0,
    "high":  16,
    "med":   28,
    "low":   36,
    "tiny":  40,
}


def lora_grad_ckpt_can_be_off(tier: str, model_class: str) -> bool:
    """Return True if `gradient_checkpointing=False` is *safe to try* at the
    given VRAM tier for this model class (batch sizes up to the tier's
    `SDXL_LORA_BATCH_CHOICES_BY_TIER` max)."""
    params_b = MODEL_PARAMS_B.get(model_class, 10.0)  # unknown → assume big
    # Rough VRAM headroom for batch=4 LoRA without ckpt:
    #   weights (bf16) + grads (bf16) + activations + optimizer state
    #   ≈ 4 × params_b GB + ~12 GB activations at 1024² bs=4
    needed_gb = 4 * params_b + 12.0
    tier_vram = {"xl": 80, "large": 48, "high": 32, "med": 24, "low": 16, "tiny": 8}
    headroom_gb = tier_vram.get(tier, 8) - needed_gb
    return headroom_gb >= 4.0  # need 4GB safety margin


def lora_grad_ckpt_baseline(tier: str, model_class: str) -> bool:
    """The default (baseline) gradient_checkpointing flag. False = faster;
    True = safer (works regardless of VRAM headroom)."""
    if not lora_grad_ckpt_can_be_off(tier, model_class):
        return True
    # If we COULD turn it off, do — small models on big cards run faster.
    return False


def lora_grad_ckpt_varies(tier: str, model_class: str) -> bool:
    """Whether the orchestrator should search both ckpt=True and ckpt=False.
    Only varies if the tier+model combo can plausibly run ckpt=False at small
    batch sizes; otherwise pinned True for safety."""
    return lora_grad_ckpt_can_be_off(tier, model_class)


# Legacy maps (kept for backwards compat with code that doesn't pass model_class).
# These assume the SDXL-sized case. New code should call the functions above.
GRAD_CKPT_VARIES_BY_TIER: dict[str, bool] = {
    "xl":    True,
    "large": True,
    "high":  True,
    "med":   False,
    "low":   False,
    "tiny":  False,
}
GRAD_CKPT_BASELINE_BY_TIER: dict[str, bool] = {
    "xl":    False,
    "large": False,
    "high":  False,
    "med":   True,
    "low":   True,
    "tiny":  True,
}
