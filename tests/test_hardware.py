from bracket.hardware import (
    SDXL_LORA_BATCH_CHOICES_BY_TIER,
    SDXL_LORA_DEFAULT_BATCH_BY_TIER,
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
