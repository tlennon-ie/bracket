"""Tests for the model+type registry."""
from __future__ import annotations

from bracket.registry import (
    PRESETS,
    SESSION_FIELDS,
    get_preset,
    list_model_families,
    training_types_for,
)


def test_each_preset_has_unique_id_and_required_fields():
    ids = [p.id for p in PRESETS]
    assert len(ids) == len(set(ids)), f"duplicate preset ids: {ids}"
    for p in PRESETS:
        # Must have at least one required field (the model itself)
        required = [f for f in p.fields if f.required]
        assert required, f"{p.id} has no required fields"


def test_list_families_unique_in_order():
    families = list_model_families()
    assert families == ["SDXL", "Z-Image", "Flux-2-Klein"]


def test_training_types_for_known_families():
    assert set(training_types_for("SDXL")) == {"LoRA", "Full FT"}
    assert set(training_types_for("Z-Image")) == {"LoRA", "Full FT"}
    # Flux-2-Klein in musubi has no full-FT script
    assert training_types_for("Flux-2-Klein") == ["LoRA"]


def test_get_preset_known_and_unknown():
    sdxl = get_preset("SDXL", "LoRA")
    assert sdxl is not None and sdxl.id == "sdxl-lora"
    assert get_preset("SDXL", "DoRA") is None
    assert get_preset("Mystery", "LoRA") is None


def test_session_fields_have_dataset_and_output():
    names = [f.name for f in SESSION_FIELDS]
    assert "dataset_toml" in names
    assert "output_dir" in names


def test_required_field_labels_marked_with_asterisk():
    for p in PRESETS:
        for f in p.fields:
            if f.required:
                assert f.label.endswith("*"), f"{p.id}.{f.name} required but no '*' on label: {f.label!r}"


def test_sdxl_lora_preset_constructs_trainer(tmp_path):
    from bracket.registry import get_preset
    sd = tmp_path / "sd-scripts"; sd.mkdir()
    (sd / "sdxl_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    py = tmp_path / "python.exe"; py.write_bytes(b"")
    preset = get_preset("SDXL", "LoRA")
    trainer = preset.trainer_factory(
        sd_scripts_dir=str(sd), venv_python=str(py),
        pretrained_model="C:/fake", vram_gb=32.0,
    )
    assert trainer.name == "sdxl-sd-scripts"


def test_zimage_preset_says_pre_cache_required():
    p = get_preset("Z-Image", "LoRA")
    assert p.needs_pre_cache is True
    p_full = get_preset("Z-Image", "Full FT")
    assert p_full.needs_pre_cache is True


def test_sdxl_preset_does_not_need_pre_cache():
    assert get_preset("SDXL", "LoRA").needs_pre_cache is False
    assert get_preset("SDXL", "Full FT").needs_pre_cache is False
