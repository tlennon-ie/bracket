"""Tests for the Flux2DevLoRATrainer adapter (musubi-tuner flux_2_train_network,
full FLUX.2-dev). We don't launch the real trainer — just verify search-space
declaration, config roundtripping, curated-config validity, and cmdline
construction (incl. the dev-vs-klein --model_version selector)."""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FloatKnob
from bracket.trainer.flux2_dev_lora import Flux2DevLoRAConfig, Flux2DevLoRATrainer


@pytest.fixture
def fake_musubi(tmp_path: Path) -> Path:
    """Pretend musubi-tuner dir with the FLUX.2 train script at the nested path."""
    d = tmp_path / "musubi-tuner"
    nested = d / "src" / "musubi_tuner"
    nested.mkdir(parents=True)
    (nested / "flux_2_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    return d


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def _make_trainer(fake_musubi: Path, fake_python: Path, vram_gb: float = 32.0) -> Flux2DevLoRATrainer:
    return Flux2DevLoRATrainer(
        musubi_dir=fake_musubi,
        venv_python=fake_python,
        dit_path="C:/fake/flux2-dev.safetensors",
        vae_path="C:/fake/ae.safetensors",
        text_encoder_path="C:/fake/mistral3-00001-of-00010.safetensors",
        vram_gb=vram_gb,
    )


def test_init_rejects_missing_musubi(tmp_path: Path, fake_python: Path):
    with pytest.raises(FileNotFoundError):
        Flux2DevLoRATrainer(
            musubi_dir=tmp_path / "nope",
            venv_python=fake_python,
            dit_path="x", vae_path="y", text_encoder_path="z",
            vram_gb=32.0,
        )


def test_init_rejects_missing_python(fake_musubi: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Flux2DevLoRATrainer(
            musubi_dir=fake_musubi,
            venv_python=tmp_path / "no_python.exe",
            dit_path="x", vae_path="y", text_encoder_path="z",
            vram_gb=32.0,
        )


def test_name_is_flux2_dev_lora(fake_musubi: Path, fake_python: Path):
    assert _make_trainer(fake_musubi, fake_python).name == "flux2-dev-lora"


def test_search_space_has_expected_knobs(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    space = t.declare_search_space()
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    assert isinstance(space.knobs["optimizer_type"], CategoricalKnob)
    assert "AdamW8bit" in space.knobs["optimizer_type"].choices
    assert isinstance(space.knobs["lr_scheduler"], CategoricalKnob)
    assert "network_dim" in space.knobs
    assert "network_alpha" in space.knobs


def test_baseline_config_validates_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


def test_curated_configs_validate_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 24.0, 32.0, 80.0):
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
    knobs = {
        "learning_rate": 5e-5, "optimizer_type": "Lion",
        "lr_scheduler": "constant", "lr_warmup_steps": 50,
        "network_dim": 16, "network_alpha": 8.0,
        "discrete_flow_shift": 3.5,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
        "max_grad_norm": 1.0,
        "fp8_base": True, "fp8_scaled": True,
        "gradient_checkpointing": True,
        "dataloader_workers": 2,
    }
    config = t.config_from_dict(knobs)
    assert config.learning_rate == 5e-5
    assert config.optimizer_type == "Lion"
    assert config.network_dim == 16
    assert config.network_alpha == 8.0


def test_prepare_run_constructs_expected_cmd(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=200, seed=42,
    )
    cmd = spec.cmd
    # accelerate launch wrapper
    assert cmd[0] == str(fake_python.resolve())
    assert "accelerate.commands.launch" in cmd
    assert "--num_cpu_threads_per_process" in cmd
    # FLUX.2 train script
    assert "flux_2_train_network.py" in cmd[-1] or any(
        c.endswith("flux_2_train_network.py") for c in cmd
    )
    # FLUX.2-dev specific selector + module
    assert "--model_version" in cmd
    assert cmd[cmd.index("--model_version") + 1] == "dev"
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora_flux_2"
    # FLUX.2 timestep sampling
    assert "--timestep_sampling" in cmd
    assert cmd[cmd.index("--timestep_sampling") + 1] == "flux2_shift"
    # weight args
    assert "--dit" in cmd
    assert cmd[cmd.index("--dit") + 1] == "C:/fake/flux2-dev.safetensors"
    assert "--vae" in cmd
    assert cmd[cmd.index("--vae") + 1] == "C:/fake/ae.safetensors"
    assert "--text_encoder" in cmd
    assert cmd[cmd.index("--text_encoder") + 1] == "C:/fake/mistral3-00001-of-00010.safetensors"
    # core training args
    assert "--max_train_steps" in cmd
    assert cmd[cmd.index("--max_train_steps") + 1] == "200"
    assert "--learning_rate" in cmd
    assert cmd[cmd.index("--learning_rate") + 1] == f"{config.learning_rate:.10g}"
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"
    # env + dirs + cwd
    assert spec.cwd == fake_musubi.resolve()
    assert "PYTHONIOENCODING" in spec.env
    assert spec.tfevents_glob.endswith("events.out.tfevents.*")
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()
    # sample dir is musubi singular default
    assert spec.sample_dir is not None
    assert spec.sample_dir.name == "sample"


def test_prepare_run_rejects_wrong_config_type(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("", encoding="utf-8")

    class Other:
        pass

    with pytest.raises(TypeError):
        t.prepare_run(
            run_dir=tmp_path / "run", config=Other(),  # type: ignore[arg-type]
            dataset_toml=dataset_toml, max_steps=10, seed=0,
        )


def test_prepare_run_omits_sample_args_when_no_prompts(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("", encoding="utf-8")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=10, seed=0,
    )
    assert "--sample_prompts" not in spec.cmd
    assert "--sample_every_n_steps" not in spec.cmd


def test_session_setup_commands_pre_cache(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    specs = t.session_setup_commands(dataset_toml=dataset_toml, run_dir=tmp_path / "run")
    assert len(specs) == 2
    joined = [" ".join(s.cmd) for s in specs]
    assert any("flux_2_cache_latents" in j for j in joined)
    assert any("flux_2_cache_text_encoder_outputs" in j for j in joined)
    # pre-cache passes the right weights
    assert any("C:/fake/ae.safetensors" in j for j in joined)
    assert any("C:/fake/mistral3-00001-of-00010.safetensors" in j for j in joined)
