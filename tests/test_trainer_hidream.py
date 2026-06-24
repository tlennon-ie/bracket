"""Tests for the HiDreamLoRATrainer adapter (musubi-tuner hidream_o1_train_network).

We never launch the real trainer — these tests verify search-space declaration,
config roundtripping, curated/baseline validation, and cmdline construction with
a fake musubi_dir + fake train script (the SDXL test pattern).

HiDream-O1 is unusual among the musubi image adapters:
  - a SINGLE model-weight argument (--dit); no --vae, no --text_encoder
    (tokenizer/processor load automatically from the official HF repos)
  - pre-cache scripts take no weight paths (hidream_o1_cache_pixel /
    hidream_o1_cache_text_encoder_outputs)
  - uniform timestep sampling + noise_scale params instead of discrete_flow_shift
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FloatKnob
from bracket.trainer.hidream_lora import HiDreamLoRAConfig, HiDreamLoRATrainer


@pytest.fixture
def fake_musubi(tmp_path: Path) -> Path:
    """Pretend musubi-tuner dir with the HiDream-O1 LoRA train script in place."""
    d = tmp_path / "musubi-tuner"
    pkg = d / "src" / "musubi_tuner"
    pkg.mkdir(parents=True)
    (pkg / "hidream_o1_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    return d


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def _make_trainer(fake_musubi: Path, fake_python: Path, vram_gb: float = 32.0) -> HiDreamLoRATrainer:
    return HiDreamLoRATrainer(
        musubi_dir=fake_musubi,
        venv_python=fake_python,
        dit_path="C:/fake/hidream_o1_image_bf16.safetensors",
        vram_gb=vram_gb,
    )


def test_init_rejects_missing_musubi(tmp_path: Path, fake_python: Path):
    with pytest.raises(FileNotFoundError):
        HiDreamLoRATrainer(
            musubi_dir=tmp_path / "nope",
            venv_python=fake_python,
            dit_path="C:/fake/dit.safetensors",
        )


def test_init_rejects_missing_python(fake_musubi: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        HiDreamLoRATrainer(
            musubi_dir=fake_musubi,
            venv_python=tmp_path / "no_python.exe",
            dit_path="C:/fake/dit.safetensors",
        )


def test_name_is_hidream_lora(fake_musubi: Path, fake_python: Path):
    assert _make_trainer(fake_musubi, fake_python).name == "hidream-lora"


def test_search_space_has_expected_knobs(fake_musubi: Path, fake_python: Path):
    space = _make_trainer(fake_musubi, fake_python).declare_search_space()
    # learning_rate is a log FloatKnob
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    # network_dim / network_alpha are categorical
    assert isinstance(space.knobs["network_dim"], CategoricalKnob)
    assert isinstance(space.knobs["network_alpha"], CategoricalKnob)
    # scheduler + optimizer categorical
    assert isinstance(space.knobs["optimizer_type"], CategoricalKnob)
    assert "AdamW8bit" in space.knobs["optimizer_type"].choices
    assert isinstance(space.knobs["lr_scheduler"], CategoricalKnob)


def test_baseline_config_validates_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


def test_curated_configs_validate_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        curated = t.curated_configs()
        assert len(curated) >= 1
        for config in curated:
            d = config.to_dict()
            for k, knob in space.knobs.items():
                if k in d:
                    knob.validate(d[k])


def test_config_from_dict_roundtrips(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    space = t.declare_search_space()
    import random

    knobs = space.sample(random.Random(0))
    config = t.config_from_dict(knobs)
    assert config.learning_rate == float(knobs["learning_rate"])
    assert config.optimizer_type == str(knobs["optimizer_type"])
    assert config.network_dim == int(knobs["network_dim"])
    assert config.network_alpha == float(knobs["network_alpha"])
    assert config.lr_scheduler == str(knobs["lr_scheduler"])


def test_prepare_run_constructs_expected_cmd(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text(
        "[[datasets]]\nresolution = [1024, 1024]\n"
        "  [[datasets.subsets]]\n  image_dir = 'C:/imgs'\n  num_repeats = 1\n",
        encoding="utf-8",
    )
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=200, seed=42,
        sample_prompts=tmp_path / "prompts.txt", sample_every_n_steps=100,
    )
    cmd = spec.cmd
    # accelerate launch wrapper
    assert cmd[0] == str(fake_python.resolve())
    assert "accelerate.commands.launch" in cmd
    assert "--num_cpu_threads_per_process" in cmd
    # musubi adapters pass the full script path; check via substring.
    assert any("hidream_o1_train_network.py" in part for part in cmd)
    # single --dit weight arg; NO --vae / --text_encoder for HiDream
    assert "--dit" in cmd
    assert cmd[cmd.index("--dit") + 1] == "C:/fake/hidream_o1_image_bf16.safetensors"
    assert "--vae" not in cmd
    assert "--text_encoder" not in cmd
    # HiDream LoRA network module
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora_hidream_o1"
    # HiDream-specific required flags
    assert "--model_type" in cmd
    assert cmd[cmd.index("--model_type") + 1] == config.model_type
    assert "--task" in cmd
    assert cmd[cmd.index("--task") + 1] == "t2i"
    assert "--timestep_sampling" in cmd
    assert cmd[cmd.index("--timestep_sampling") + 1] == "uniform"
    assert "--noise_scale_start" in cmd
    assert "--noise_scale_end" in cmd
    assert "--noise_clip_std" in cmd
    # core training args
    assert "--max_train_steps" in cmd
    assert cmd[cmd.index("--max_train_steps") + 1] == "200"
    assert "--learning_rate" in cmd
    assert cmd[cmd.index("--learning_rate") + 1] == f"{config.learning_rate:.10g}"
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert "--output_dir" in cmd
    assert cmd[cmd.index("--output_dir") + 1] == str(spec.output_dir)
    # sample args present
    assert "--sample_prompts" in cmd
    assert "--sample_every_n_steps" in cmd
    # paths/env
    assert spec.cwd == fake_musubi.resolve()
    assert "PYTHONIOENCODING" in spec.env
    assert spec.tfevents_glob.endswith("events.out.tfevents.*")
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()
    # sample dir is singular per musubi convention
    assert spec.sample_dir == spec.output_dir / "sample"


def test_session_setup_commands_use_hidream_cache_scripts(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text(
        "[[datasets]]\nresolution = [1024, 1024]\n"
        "  [[datasets.subsets]]\n  image_dir = 'C:/imgs'\n  num_repeats = 1\n",
        encoding="utf-8",
    )
    specs = t.session_setup_commands(dataset_toml=dataset_toml, run_dir=tmp_path / "run")
    assert len(specs) == 2
    pixel_cmd = " ".join(specs[0].cmd)
    te_cmd = " ".join(specs[1].cmd)
    assert "hidream_o1_cache_pixel" in pixel_cmd
    assert "hidream_o1_cache_text_encoder_outputs" in te_cmd
    # HiDream cache scripts take NO --vae / --text_encoder weight paths
    for spec in specs:
        assert "--vae" not in spec.cmd
        assert "--text_encoder" not in spec.cmd
        assert "--dataset_config" in spec.cmd


def test_prepare_run_omits_sample_args_when_none(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text(
        "[[datasets]]\nresolution = [1024, 1024]\n"
        "  [[datasets.subsets]]\n  image_dir = 'C:/imgs'\n  num_repeats = 1\n",
        encoding="utf-8",
    )
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=10, seed=0,
    )
    assert "--sample_prompts" not in spec.cmd
    assert "--sample_every_n_steps" not in spec.cmd
