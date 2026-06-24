"""Tests for the QwenImageEditLoRATrainer adapter.

We never launch musubi-tuner — just verify cmdline construction (esp. the
``--model_version`` flag and the unified ``qwen_image_train_network.py`` script),
the model-version validation, config roundtripping, and the pre-cache commands
threading ``--model_version`` into BOTH cache scripts.

Background: upstream musubi-tuner unified the Qwen-Image-Edit path onto the plain
Qwen-Image scripts and selects the architecture with ``--model_version``
(edit / edit-2509 / edit-2511). The separate ``qwen_image_edit_*`` scripts no
longer exist; the default ``original`` would silently train the wrong arch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FixedKnob, FloatKnob
from bracket.trainer.qwen_image_edit_lora import (
    QwenImageEditLoRAConfig,
    QwenImageEditLoRATrainer,
)


@pytest.fixture
def fake_musubi(tmp_path: Path) -> Path:
    """Pretend musubi-tuner dir with the UNIFIED train script (no edit-specific)."""
    d = tmp_path / "musubi-tuner"
    pkg = d / "src" / "musubi_tuner"
    pkg.mkdir(parents=True)
    (pkg / "qwen_image_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    return d


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def _make_trainer(
    fake_musubi: Path,
    fake_python: Path,
    *,
    model_version: str = "edit-2509",
    vram_gb: float = 32.0,
) -> QwenImageEditLoRATrainer:
    return QwenImageEditLoRATrainer(
        musubi_dir=fake_musubi,
        venv_python=fake_python,
        dit_path="C:/fake/qwen_image_edit_2509_dit.safetensors",
        vae_path="C:/fake/qwen_image_vae.safetensors",
        text_encoder_path="C:/fake/qwen2.5-vl-7b",
        model_version=model_version,
        vram_gb=vram_gb,
    )


def test_init_uses_unified_train_script(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    # The adapter targets the unified script, NOT a (now nonexistent)
    # qwen_image_edit_train_network.py.
    assert t.train_script.name == "qwen_image_train_network.py"
    assert t.train_script.exists()


def test_init_rejects_invalid_model_version(fake_musubi: Path, fake_python: Path):
    with pytest.raises(ValueError):
        _make_trainer(fake_musubi, fake_python, model_version="original")
    with pytest.raises(ValueError):
        _make_trainer(fake_musubi, fake_python, model_version="layered")


def test_init_accepts_all_edit_versions(fake_musubi: Path, fake_python: Path):
    for ver in ("edit", "edit-2509", "edit-2511"):
        t = _make_trainer(fake_musubi, fake_python, model_version=ver)
        assert t.model_version == ver


def test_init_rejects_missing_train_script(tmp_path: Path, fake_python: Path):
    empty = tmp_path / "musubi-empty"
    (empty / "src" / "musubi_tuner").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        QwenImageEditLoRATrainer(
            musubi_dir=empty,
            venv_python=fake_python,
            dit_path="C:/fake/dit",
            vae_path="C:/fake/vae",
            text_encoder_path="C:/fake/te",
        )


def test_search_space_has_expected_knobs(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    space = t.declare_search_space()
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    assert isinstance(space.knobs["optimizer_type"], CategoricalKnob)
    assert isinstance(space.knobs["discrete_flow_shift"], FloatKnob)
    assert isinstance(space.knobs["fp8_base"], FixedKnob)
    assert isinstance(space.knobs["blocks_to_swap"], FixedKnob)


def test_baseline_config_validates_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


def test_config_from_dict_roundtrips(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    knobs = {
        "learning_rate": 5e-5, "optimizer_type": "Lion",
        "lr_scheduler": "constant", "lr_warmup_steps": 50,
        "network_dim": 16, "network_alpha": 8.0,
        "discrete_flow_shift": 6.0, "train_batch_size": 1,
        "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
        "max_grad_norm": 1.0, "fp8_base": True, "fp8_scaled": True,
        "fp8_vl": False, "gradient_checkpointing": False,
        "blocks_to_swap": 0, "dataloader_workers": 2,
    }
    config = t.config_from_dict(knobs)
    assert isinstance(config, QwenImageEditLoRAConfig)
    assert config.learning_rate == 5e-5
    assert config.optimizer_type == "Lion"
    assert config.discrete_flow_shift == 6.0


def test_prepare_run_passes_model_version(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python, model_version="edit-2511")
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=200, seed=42,
    )
    cmd = spec.cmd
    assert any("qwen_image_train_network.py" in part for part in cmd)
    assert "--model_version" in cmd
    assert cmd[cmd.index("--model_version") + 1] == "edit-2511"
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora_qwen_image"
    assert "--dit" in cmd
    assert cmd[cmd.index("--dit") + 1] == "C:/fake/qwen_image_edit_2509_dit.safetensors"


def test_prepare_run_rejects_wrong_config_type(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    with pytest.raises(TypeError):
        t.prepare_run(
            run_dir=tmp_path / "run", config=object(),  # type: ignore[arg-type]
            dataset_toml=dataset_toml, max_steps=10, seed=0,
        )


def test_session_setup_uses_unified_cache_with_model_version(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python, model_version="edit-2509")
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    specs = t.session_setup_commands(
        dataset_toml=dataset_toml, run_dir=tmp_path / "run",
    )
    assert len(specs) == 2
    latents_cmd, te_cmd = specs[0].cmd, specs[1].cmd
    # Unified (NOT edit-specific) cache modules.
    assert "musubi_tuner.qwen_image_cache_latents" in latents_cmd
    assert "musubi_tuner.qwen_image_cache_text_encoder_outputs" in te_cmd
    # BOTH cache commands must carry the model_version so the edit caches
    # encode control images + edit-aware text-encoder prompts.
    for cmd in (latents_cmd, te_cmd):
        assert "--model_version" in cmd
        assert cmd[cmd.index("--model_version") + 1] == "edit-2509"
    assert specs[0].tfevents_glob == ""
    assert specs[1].tfevents_glob == ""
