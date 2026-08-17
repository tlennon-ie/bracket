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
        "SD3.5", "HunyuanVideo", "Wan 2.2", "Wan 2.1", "FramePack",
        "FLUX.2", "HiDream", "HunyuanVideo 1.5", "Kandinsky 5",
        "Ideogram 4", "Krea 2",
        # ai-toolkit (ostris) families
        "Chroma", "Lumina2", "OmniGen2", "Flex.1", "Flex.2",
    }
    assert expected_extension.issubset(set(families))
    # LTX-Video (musubi) was dropped — native LTX-2 covers it now.
    assert "LTX-Video" not in families


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
    # Wan has no full-FT script in musubi — LoRA only for both versions.
    assert training_types_for("Wan 2.2") == ["LoRA"]
    assert training_types_for("Wan 2.1") == ["LoRA"]
    # LTX-Video (musubi) dropped — superseded by the native LTX-2 presets.
    assert training_types_for("LTX-Video") == []
    assert training_types_for("FramePack") == ["LoRA"]
    # Newly-wired musubi families (LoRA-only)
    assert training_types_for("FLUX.2") == ["LoRA"]
    assert training_types_for("HiDream") == ["LoRA"]
    assert training_types_for("HunyuanVideo 1.5") == ["LoRA"]
    assert training_types_for("Kandinsky 5") == ["LoRA"]


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


def _stub_sd_scripts(tmp_path, script_name: str):
    """Create a stub sd-scripts directory with a no-op training script."""
    sd = tmp_path / "sd-scripts"
    sd.mkdir()
    (sd / script_name).write_text("# placeholder\n", encoding="utf-8")
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    return sd, py


def test_flux1_lora_preset_constructs_trainer(tmp_path):
    sd, py = _stub_sd_scripts(tmp_path, "flux_train_network.py")
    preset = get_preset("Flux.1", "LoRA")
    assert preset.needs_pre_cache is False
    trainer = preset.trainer_factory(
        sd_scripts_dir=str(sd), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/ae",
        t5xxl_path="/x/t5", clip_l_path="/x/clip",
        vram_gb=32.0,
    )
    assert trainer.name == "flux1-lora-sd-scripts"


def test_flux1_full_preset_constructs_trainer(tmp_path):
    sd, py = _stub_sd_scripts(tmp_path, "flux_train.py")
    preset = get_preset("Flux.1", "Full FT")
    assert preset.needs_pre_cache is False
    trainer = preset.trainer_factory(
        sd_scripts_dir=str(sd), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/ae",
        t5xxl_path="/x/t5", clip_l_path="/x/clip",
        vram_gb=32.0,
    )
    assert trainer.name == "flux1-full-sd-scripts"


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


def test_ltx2_presets_registered_and_ltx_video_removed():
    ids = [p.id for p in PRESETS]
    assert "ltx2-t2v-lora" in ids
    assert "ltx2-i2v-lora" in ids
    # The native LTX-2 family superseded the dropped musubi "LTX-Video" preset.
    families = list_model_families()
    assert "LTX-2" in families
    assert "LTX-Video" not in families
    assert "ltx-video-lora" not in ids


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


def test_hidream_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "hidream_o1_train_network.py")
    preset = get_preset("HiDream", "LoRA")
    assert preset.needs_pre_cache is True
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vram_gb=32.0,
    )
    assert trainer.name == "hidream-lora"


def test_hunyuan_video_15_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "hv_1_5_train_network.py")
    preset = get_preset("HunyuanVideo 1.5", "LoRA")
    assert preset.needs_pre_cache is True
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae",
        text_encoder_path="/x/qwen", byt5_path="/x/byt5",
        vram_gb=32.0,
    )
    assert trainer.name == "hunyuan-video-15-lora"


def test_flux2_dev_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "flux_2_train_network.py")
    preset = get_preset("FLUX.2", "LoRA")
    assert preset.needs_pre_cache is True
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/ae",
        text_encoder_path="/x/mistral", vram_gb=32.0,
    )
    assert trainer.name == "flux2-dev-lora"


def test_kandinsky5_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "kandinsky5_train_network.py")
    preset = get_preset("Kandinsky 5", "LoRA")
    assert preset.needs_pre_cache is True
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae",
        text_encoder_qwen_path="/x/qwen", text_encoder_clip_path="/x/clip",
        vram_gb=32.0,
    )
    assert trainer.name == "kandinsky5-lora"


def test_ideogram4_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "ideogram4_train_network.py")
    preset = get_preset("Ideogram 4", "LoRA")
    assert preset.needs_pre_cache is True
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae", text_encoder_path="/x/te",
        vram_gb=24.0,
    )
    assert trainer.name == "ideogram4-lora"


def test_krea2_lora_preset_constructs_trainer(tmp_path):
    musubi, py = _stub_musubi(tmp_path, "krea2_train_network.py")
    preset = get_preset("Krea 2", "LoRA")
    assert preset.needs_pre_cache is True
    trainer = preset.trainer_factory(
        musubi_dir=str(musubi), venv_python=str(py),
        dit_path="/x/dit", vae_path="/x/vae", text_encoder_path="/x/te",
        vram_gb=24.0,
    )
    assert trainer.name == "krea2-lora"


# ─────────────────────────── ai-toolkit (ostris) ───────────────────────────


def _stub_aitk(tmp_path):
    """Create a stub ai-toolkit checkout dir with a no-op run.py."""
    d = tmp_path / "ai-toolkit"
    d.mkdir()
    (d / "run.py").write_text("# placeholder\n", encoding="utf-8")
    py = tmp_path / "python.exe"
    py.write_bytes(b"")
    return d, py


def test_aitk_presets_registered():
    ids = [p.id for p in PRESETS]
    for pid in (
        "aitk-chroma-lora", "aitk-lumina2-lora", "aitk-omnigen2-lora",
        "aitk-flex1-lora", "aitk-flex2-lora",
    ):
        assert pid in ids, f"missing ai-toolkit preset {pid}"


def test_aitk_presets_do_not_need_pre_cache():
    for fam in ("Chroma", "Lumina2", "OmniGen2", "Flex.1", "Flex.2"):
        p = get_preset(fam, "LoRA")
        assert p is not None and p.needs_pre_cache is False


def test_aitk_chroma_lora_preset_constructs_trainer(tmp_path):
    d, py = _stub_aitk(tmp_path)
    preset = get_preset("Chroma", "LoRA")
    trainer = preset.trainer_factory(
        aitk_dir=str(d), venv_python=str(py),
        model_name_or_path="lodestones/Chroma", vram_gb=24.0,
    )
    assert trainer.name == "aitk-lora-chroma"
    assert trainer.model_id == "chroma"
    assert trainer.model_extra == {"arch": "chroma", "quantize": True}


def test_aitk_omnigen2_lora_preset_constructs_trainer(tmp_path):
    d, py = _stub_aitk(tmp_path)
    preset = get_preset("OmniGen2", "LoRA")
    trainer = preset.trainer_factory(
        aitk_dir=str(d), venv_python=str(py),
        model_name_or_path="OmniGen2/OmniGen2", vram_gb=24.0,
    )
    assert trainer.name == "aitk-lora-omnigen2"
    assert trainer.model_id == "omnigen2"
    assert trainer.model_extra == {"arch": "omnigen2", "quantize_te": True}


def test_aitk_flex_presets_construct_trainers(tmp_path):
    d, py = _stub_aitk(tmp_path)
    flex1 = get_preset("Flex.1", "LoRA").trainer_factory(
        aitk_dir=str(d), venv_python=str(py),
        model_name_or_path="ostris/Flex.1-alpha", vram_gb=24.0,
    )
    assert flex1.name == "aitk-lora-flex1"
    flex2 = get_preset("Flex.2", "LoRA").trainer_factory(
        aitk_dir=str(d), venv_python=str(py),
        model_name_or_path="ostris/Flex.2-preview", vram_gb=24.0,
    )
    assert flex2.name == "aitk-lora-flex2"


def test_aitk_default_model_name_or_path_per_preset():
    expected = {
        "Chroma": "lodestones/Chroma",
        "Lumina2": "Alpha-VLLM/Lumina-Image-2.0",
        "OmniGen2": "OmniGen2/OmniGen2",
        "Flex.1": "ostris/Flex.1-alpha",
        "Flex.2": "ostris/Flex.2-preview",
    }
    for fam, hf_id in expected.items():
        p = get_preset(fam, "LoRA")
        field = next(f for f in p.fields if f.name == "model_name_or_path")
        assert field.default == hf_id


# ──────────── ai-toolkit · video + audio (LTX-2.5, MiniMax-H3, ACE-Step) ────────────


_AITK_MEDIA_PRESETS = {
    "aitk-ltx25-lora": ("LTX-2.5", "ltx25"),
    "aitk-minimax-h3-lora": ("MiniMax-H3", "minimax_h3"),
    "aitk-minimax-h3-ref2va-lora": ("MiniMax-H3 Ref2VA", "minimax_h3_ref2va"),
    "aitk-ace-step-15-lora": ("ACE-Step 1.5", "ace_step_15"),
    "aitk-ace-step-15-xl-lora": ("ACE-Step 1.5 XL", "ace_step_15_xl"),
}


def test_aitk_media_presets_registered():
    ids = [p.id for p in PRESETS]
    families = list_model_families()
    for pid, (family, _profile_id) in _AITK_MEDIA_PRESETS.items():
        assert pid in ids, f"missing ai-toolkit preset {pid}"
        assert family in families, f"missing family {family}"
        assert training_types_for(family) == ["LoRA"]


def test_aitk_media_presets_construct_trainers(tmp_path):
    """Each preset wires its profile through to the adapter."""
    d, py = _stub_aitk(tmp_path)
    for pid, (family, profile_id) in _AITK_MEDIA_PRESETS.items():
        preset = get_preset(family, "LoRA")
        assert preset.id == pid
        # ai-toolkit caches latents inline — never a separate pre-cache stage.
        assert preset.needs_pre_cache is False
        trainer = preset.trainer_factory(
            aitk_dir=str(d), venv_python=str(py),
            model_name_or_path="some/model", vram_gb=32.0,
        )
        assert trainer.name == f"aitk-lora-{profile_id}"


def test_aitk_media_preset_defaults_come_from_profiles():
    """The model field's default is the profile's, not a second hardcoding."""
    from bracket.trainer.aitk_profiles import get_profile

    for _pid, (family, profile_id) in _AITK_MEDIA_PRESETS.items():
        preset = get_preset(family, "LoRA")
        field = next(f for f in preset.fields if f.name == "model_name_or_path")
        assert field.default == get_profile(profile_id).default_model
        assert field.required is True


def test_every_aitk_preset_resolves_a_real_profile(tmp_path):
    """No ai-toolkit preset may reference a profile id that does not exist.

    ``_build_aitk_lora`` raises KeyError on an unknown id; constructing every
    ai-toolkit preset here turns a typo into a test failure instead of a
    runtime error the user only sees when they pick that model.
    """
    d, py = _stub_aitk(tmp_path)
    aitk_presets = [p for p in PRESETS if p.id.startswith("aitk-")]
    assert len(aitk_presets) == 10, [p.id for p in aitk_presets]
    for preset in aitk_presets:
        trainer = preset.trainer_factory(
            aitk_dir=str(d), venv_python=str(py),
            model_name_or_path="some/model", vram_gb=24.0,
        )
        assert trainer.name.startswith("aitk-lora-")
