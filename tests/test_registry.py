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


def test_list_families_includes_all_supported():
    families = list_model_families()
    # The first three are the original v0.1 set; later additions extend.
    assert families[:3] == ["SDXL", "Z-Image", "Flux-2-Klein"]
    expected_extension = {
        "Flux.1", "Flux.1-Kontext", "Qwen-Image", "Qwen-Image-Edit",
        "SD3.5", "HunyuanVideo", "Wan 2.2", "Wan 2.1", "LTX-Video", "FramePack",
    }
    assert expected_extension.issubset(set(families))


def test_training_types_for_known_families():
    assert set(training_types_for("SDXL")) == {"LoRA", "Full FT"}
    assert set(training_types_for("Z-Image")) == {"LoRA", "Full FT"}
    # Flux-2-Klein in musubi has no full-FT script
    assert training_types_for("Flux-2-Klein") == ["LoRA"]
    # Newer families
    assert set(training_types_for("Flux.1")) == {"LoRA", "Full FT"}
    assert training_types_for("Flux.1-Kontext") == ["LoRA"]
    assert set(training_types_for("Qwen-Image")) == {"LoRA", "Full FT"}
    assert training_types_for("Qwen-Image-Edit") == ["LoRA"]
    assert set(training_types_for("SD3.5")) == {"LoRA", "Full FT"}
    assert set(training_types_for("HunyuanVideo")) == {"LoRA", "Full FT"}
    assert set(training_types_for("Wan 2.2")) == {"LoRA", "Full FT"}
    assert training_types_for("Wan 2.1") == ["LoRA"]
    assert training_types_for("LTX-Video") == ["LoRA"]
    assert training_types_for("FramePack") == ["LoRA"]


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


def _stub_musubi(tmp_path, script_name: str):
    """Create a stub musubi-tuner directory with a no-op training script."""
    musubi = tmp_path / "musubi-tuner"
    pkg = musubi / "src" / "musubi_tuner"
    pkg.mkdir(parents=True)
    (pkg / script_name).write_text("# placeholder\n", encoding="utf-8")
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    return musubi, py


def test_qwen_image_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "qwen_image_train_network.py")
    preset = get_preset("Qwen-Image", "LoRA")
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae",
        text_encoder_path="/x/te", vram_gb=32.0,
    )
    assert trainer.name == "qwen-image-lora-musubi"


def test_flux1_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "flux_train_network.py")
    preset = get_preset("Flux.1", "LoRA")
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/ae",
        t5xxl_path="/x/t5", clip_l_path="/x/clip",
        vram_gb=32.0,
    )
    assert trainer.name == "flux1-lora-musubi"


def test_hunyuan_video_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "hv_train_network.py")
    preset = get_preset("HunyuanVideo", "LoRA")
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae",
        text_encoder1_path="/x/llama", text_encoder2_path="/x/clip",
        vram_gb=32.0,
    )
    assert trainer.name == "hunyuan-video-lora-musubi"


def test_wan22_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "wan_train_network.py")
    preset = get_preset("Wan 2.2", "LoRA")
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae",
        text_encoder_path="/x/umt5", vram_gb=32.0,
    )
    assert trainer.name == "wan-lora-musubi"
    assert trainer.wan_version == "2.2"


def test_ltx_video_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "ltxv_train_network.py")
    preset = get_preset("LTX-Video", "LoRA")
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae",
        text_encoder_path="/x/t5", vram_gb=32.0,
    )
    assert trainer.name == "ltx-video-lora-musubi"


def test_framepack_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "fpack_train_network.py")
    preset = get_preset("FramePack", "LoRA")
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae",
        text_encoder1_path="/x/llama", text_encoder2_path="/x/clip",
        vram_gb=32.0,
    )
    assert trainer.name == "framepack-lora-musubi"


def _stub_ltx_trainer(tmp_path):
    """Create a stub native ltx-trainer dir with the expected scripts."""
    d = tmp_path / "ltx-trainer"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "train.py").write_text("# placeholder\n", encoding="utf-8")
    (d / "scripts" / "process_dataset.py").write_text("# placeholder\n", encoding="utf-8")
    return d


def test_ltx2_presets_registered_and_distinct_from_ltx_video():
    ids = [p.id for p in PRESETS]
    assert "ltx2-t2v-lora" in ids
    assert "ltx2-i2v-lora" in ids
    # The native LTX-2 family is distinct from the musubi "LTX-Video" preset.
    families = list_model_families()
    assert "LTX-2" in families
    assert "LTX-Video" in families


def test_ltx2_t2v_lora_preset_constructs_trainer(tmp_path):
    d = _stub_ltx_trainer(tmp_path)
    preset = next(p for p in PRESETS if p.id == "ltx2-t2v-lora")
    assert preset.needs_pre_cache is True
    trainer = preset.trainer_factory(
        trainer_dir=str(d), model_path="/x/model",
        text_encoder_path="/x/gemma", vram_gb=32.0,
    )
    assert trainer.name == "ltx2-lora-t2v"


def test_ltx2_i2v_lora_preset_constructs_trainer(tmp_path):
    d = _stub_ltx_trainer(tmp_path)
    preset = next(p for p in PRESETS if p.id == "ltx2-i2v-lora")
    trainer = preset.trainer_factory(
        trainer_dir=str(d), model_path="/x/model",
        text_encoder_path="/x/gemma", vram_gb=32.0,
    )
    assert trainer.name == "ltx2-lora-i2v"


def test_sd35_lora_preset_constructs_trainer(tmp_path):
    sd = tmp_path / "sd-scripts"
    sd.mkdir()
    (sd / "sd3_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    preset = get_preset("SD3.5", "LoRA")
    trainer = preset.trainer_factory(
        sd_scripts_dir=str(sd), venv_python=str(py),
        pretrained_model="/x/sd3.5", vram_gb=32.0,
    )
    assert trainer.name == "sd35-lora-sd-scripts"
