import subprocess

import pytest

from bracket import hardware
from bracket.hardware import (
    SDXL_LORA_BATCH_CHOICES_BY_TIER,
    SDXL_LORA_DEFAULT_BATCH_BY_TIER,
    detect_gpu,
    vram_tier,
)


def test_vram_tier_buckets():
    # Real cards we care about land in the right tiers
    assert vram_tier(80.0) == "xl"     # H100 / A100 80GB
    assert vram_tier(48.0) == "large"  # A6000 / RTX 6000 Ada
    assert vram_tier(32.0) == "high"   # RTX 5090
    assert vram_tier(31.84) == "high"  # RTX 5090 actual reading
    assert vram_tier(24.0) == "med"    # RTX 4090 / 3090
    assert vram_tier(16.0) == "low"    # RTX 4060 Ti 16GB
    assert vram_tier(12.0) == "low"    # RTX 4070 / 3060 12GB
    assert vram_tier(8.0) == "tiny"
    assert vram_tier(6.0) == "tiny"


def test_batch_choices_monotone_in_tier():
    """Higher VRAM tier ≥ at least as much max batch_size as the tier below."""
    order = ["tiny", "low", "med", "high", "large", "xl"]
    last_max = 0
    for tier in order:
        m = max(SDXL_LORA_BATCH_CHOICES_BY_TIER[tier])
        assert m >= last_max, f"tier {tier} max={m} < previous {last_max}"
        last_max = m


def test_default_batch_inside_choices():
    for tier, choices in SDXL_LORA_BATCH_CHOICES_BY_TIER.items():
        default = SDXL_LORA_DEFAULT_BATCH_BY_TIER[tier]
        assert default in choices, f"tier {tier} default {default} not in {choices}"


# ─── detect_gpu fallback ────────────────────────────────────────────


def test_detect_gpu_returns_none_when_torch_and_nvsmi_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a CPU-only env with no torch and no nvidia-smi, detect_gpu must
    return None silently — no traceback in the log."""

    monkeypatch.setattr(hardware, "_detect_via_torch", lambda _i: None)
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: None)
    assert detect_gpu() is None


def test_detect_gpu_falls_back_to_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """When torch is unavailable (Bracket app venv) but nvidia-smi is on
    PATH, detect_gpu must shell out and return parsed info."""

    fake_nvsmi = tmp_path / "nvidia-smi"
    fake_nvsmi.write_text("")

    class _R:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 5090, 32607\n"
        stderr = ""

    monkeypatch.setattr(hardware, "_detect_via_torch", lambda _i: None)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: str(fake_nvsmi) if name == "nvidia-smi" else None)
    monkeypatch.setattr(hardware.subprocess, "run", lambda *a, **k: _R())

    info = detect_gpu()
    assert info is not None
    assert info.name == "NVIDIA GeForce RTX 5090"
    # 32607 MiB / 1024 ≈ 31.84 GiB — should land in the "high" tier
    assert 31.0 < info.total_vram_gb < 32.5
    assert vram_tier(info.total_vram_gb) == "high"


def test_detect_gpu_handles_nvidia_smi_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """If nvidia-smi exists but errors out (driver mismatch, no CUDA), we
    swallow it and return None."""

    fake_nvsmi = tmp_path / "nvidia-smi"
    fake_nvsmi.write_text("")

    def _boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    monkeypatch.setattr(hardware, "_detect_via_torch", lambda _i: None)
    monkeypatch.setattr(hardware.shutil, "which", lambda _name: str(fake_nvsmi))
    monkeypatch.setattr(hardware.subprocess, "run", _boom)
    assert detect_gpu() is None
