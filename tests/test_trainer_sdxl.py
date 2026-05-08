"""Tests for the SDXLTrainer adapter. We don't actually launch sd-scripts —
just verify cmdline construction, search space declaration, and config
roundtripping."""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FloatKnob, IntKnob, FixedKnob
from bracket.trainer.sdxl import SDXLConfig, SDXLTrainer


@pytest.fixture
def fake_sd_scripts(tmp_path: Path) -> Path:
    """Pretend sd-scripts dir with the expected entrypoint script."""
    d = tmp_path / "sd-scripts"
    d.mkdir()
    (d / "sdxl_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    return d


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def test_init_rejects_missing_sd_scripts(tmp_path: Path, fake_python: Path):
    with pytest.raises(FileNotFoundError):
        SDXLTrainer(sd_scripts_dir=tmp_path / "nope", venv_python=fake_python)


def test_init_rejects_missing_python(fake_sd_scripts: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=tmp_path / "no_python.exe")


def test_search_space_has_expected_knobs(fake_sd_scripts: Path, fake_python: Path):
    t = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=32.0)
    space = t.declare_search_space()
    assert "learning_rate" in space.knobs
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    assert isinstance(space.knobs["optimizer_type"], CategoricalKnob)
    assert "AdamW8bit" in space.knobs["optimizer_type"].choices
    assert isinstance(space.knobs["lr_warmup_steps"], IntKnob)


def test_search_space_batch_size_scales_with_vram(fake_sd_scripts: Path, fake_python: Path):
    t_low = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=8.0)
    t_high = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=32.0)
    low_choices = t_low.declare_search_space().knobs["train_batch_size"].choices
    high_choices = t_high.declare_search_space().knobs["train_batch_size"].choices
    assert max(high_choices) > max(low_choices)
    assert max(low_choices) == 1     # tiny tier pins to 1
    assert max(high_choices) >= 8    # high tier (5090) goes up to 8


def test_baseline_uses_tier_appropriate_batch_size(fake_sd_scripts: Path, fake_python: Path):
    t_low = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=8.0)
    t_high = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=32.0)
    assert t_low.baseline_config().train_batch_size == 1
    assert t_high.baseline_config().train_batch_size >= 4  # high tier baseline isn't 1


def test_baseline_config_validates_against_search_space(fake_sd_scripts: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


def test_config_from_dict_roundtrips(fake_sd_scripts: Path, fake_python: Path):
    t = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=32.0)
    knobs = {
        "learning_rate": 5e-5, "optimizer_type": "Lion",
        "lr_scheduler": "constant", "lr_warmup_steps": 50,
        "network_dim": 16, "network_alpha": 8.0,
        "noise_offset": 0.05, "train_batch_size": 1,
        "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
        "max_grad_norm": 1.0,
        "gradient_checkpointing": False,
        "dataloader_workers": 6,
    }
    config = t.config_from_dict(knobs)
    assert config.learning_rate == 5e-5
    assert config.optimizer_type == "Lion"
    assert config.network_dim == 16


def test_prepare_run_constructs_expected_cmd(
    fake_sd_scripts: Path, fake_python: Path, tmp_path: Path,
):
    t = SDXLTrainer(
        sd_scripts_dir=fake_sd_scripts, venv_python=fake_python,
        pretrained_model="C:/fake/sdxl", vram_gb=32.0,
    )
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=200, seed=42,
    )
    cmd = spec.cmd
    assert cmd[0] == str(fake_python.resolve())
    # Wrapped via `accelerate launch` — see make_accelerate_launch_prefix
    assert "accelerate.commands.launch" in cmd
    assert "--num_cpu_threads_per_process" in cmd
    assert "sdxl_train_network.py" in cmd
    assert "--max_train_steps" in cmd
    assert cmd[cmd.index("--max_train_steps") + 1] == "200"
    assert "--learning_rate" in cmd
    assert cmd[cmd.index("--learning_rate") + 1] == f"{config.learning_rate:.10g}"
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora"
    assert "--network_train_unet_only" in cmd  # required when caching TE outputs
    assert spec.cwd == fake_sd_scripts.resolve()
    assert "PYTHONIOENCODING" in spec.env
    assert spec.tfevents_glob.endswith("events.out.tfevents.*")
    # Output directories were created
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()


def test_prepare_run_omits_optional_args_when_none(
    fake_sd_scripts: Path, fake_python: Path, tmp_path: Path,
):
    t = SDXLTrainer(sd_scripts_dir=fake_sd_scripts, venv_python=fake_python, vram_gb=32.0)
    config = SDXLConfig(noise_offset=0.0, min_snr_gamma=None)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("", encoding="utf-8")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=10, seed=0,
    )
    assert "--noise_offset" not in spec.cmd
    assert "--min_snr_gamma" not in spec.cmd
