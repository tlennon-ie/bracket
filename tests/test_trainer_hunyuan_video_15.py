"""Tests for the HunyuanVideo15LoRATrainer adapter (musubi-tuner
hv_1_5_train_network). We never launch the real trainer — just verify
cmdline construction, search-space declaration, pre-cache commands, and
config roundtripping, mirroring test_trainer_sdxl.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FixedKnob, FloatKnob, IntKnob
from bracket.trainer.hunyuan_video_15_lora import (
    HunyuanVideo15LoRAConfig, HunyuanVideo15LoRATrainer,
)


@pytest.fixture
def fake_musubi(tmp_path: Path) -> Path:
    """Pretend musubi-tuner dir with the expected hv_1_5 train script."""
    d = tmp_path / "musubi-tuner"
    pkg = d / "src" / "musubi_tuner"
    pkg.mkdir(parents=True)
    (pkg / "hv_1_5_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    return d


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def _make_trainer(
    fake_musubi: Path, fake_python: Path, *, vram_gb: float = 32.0,
) -> HunyuanVideo15LoRATrainer:
    return HunyuanVideo15LoRATrainer(
        musubi_dir=fake_musubi,
        venv_python=fake_python,
        dit_path="C:/fake/hv15_dit.safetensors",
        vae_path="C:/fake/hv15_vae.safetensors",
        text_encoder_path="C:/fake/qwen25vl",
        byt5_path="C:/fake/byt5",
        vram_gb=vram_gb,
    )


def test_init_rejects_missing_musubi(tmp_path: Path, fake_python: Path):
    with pytest.raises(FileNotFoundError):
        HunyuanVideo15LoRATrainer(
            musubi_dir=tmp_path / "nope", venv_python=fake_python,
            dit_path="d", vae_path="v", text_encoder_path="te", byt5_path="b",
        )


def test_init_rejects_missing_python(fake_musubi: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        HunyuanVideo15LoRATrainer(
            musubi_dir=fake_musubi, venv_python=tmp_path / "no_python.exe",
            dit_path="d", vae_path="v", text_encoder_path="te", byt5_path="b",
        )


def test_init_rejects_bad_task(fake_musubi: Path, fake_python: Path):
    with pytest.raises(ValueError):
        HunyuanVideo15LoRATrainer(
            musubi_dir=fake_musubi, venv_python=fake_python,
            dit_path="d", vae_path="v", text_encoder_path="te", byt5_path="b",
            task="bogus",
        )


def test_search_space_has_expected_knobs(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    space = t.declare_search_space()
    assert "learning_rate" in space.knobs
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    assert isinstance(space.knobs["optimizer_type"], CategoricalKnob)
    assert "AdamW8bit" in space.knobs["optimizer_type"].choices
    assert isinstance(space.knobs["lr_scheduler"], CategoricalKnob)
    assert isinstance(space.knobs["lr_warmup_steps"], IntKnob)
    assert isinstance(space.knobs["network_dim"], CategoricalKnob)
    assert isinstance(space.knobs["network_alpha"], CategoricalKnob)
    # Non-loss-bearing knobs are pinned.
    assert isinstance(space.knobs["mixed_precision"], FixedKnob)
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


def test_curated_configs_validate_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 24.0, 32.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        curated = t.curated_configs()
        assert len(curated) >= 1
        for cfg in curated:
            d = cfg.to_dict()
            for k, knob in space.knobs.items():
                if k in d:
                    knob.validate(d[k])


def test_config_from_dict_roundtrips(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    knobs = {
        "learning_rate": 8e-5, "optimizer_type": "Lion",
        "lr_scheduler": "constant", "lr_warmup_steps": 25,
        "network_dim": 64, "network_alpha": 32.0,
        "discrete_flow_shift": 3.0,
        "train_batch_size": 1, "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16", "max_grad_norm": 1.0,
        "fp8_base": True, "fp8_vl": True,
        "gradient_checkpointing": True, "blocks_to_swap": 16,
        "dataloader_workers": 2,
    }
    config = t.config_from_dict(knobs)
    assert config.learning_rate == 8e-5
    assert config.optimizer_type == "Lion"
    assert config.network_dim == 64
    assert config.network_alpha == 32.0
    assert config.discrete_flow_shift == 3.0


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
    assert cmd[0] == str(fake_python.resolve())
    # Wrapped via `accelerate launch` — see make_accelerate_launch_prefix
    assert "accelerate.commands.launch" in cmd
    assert "--num_cpu_threads_per_process" in cmd
    # 1.5-specific train script
    assert any("hv_1_5_train_network.py" in c for c in cmd)
    # 1.5-specific network module + weights
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora_hv_1_5"
    assert "--dit" in cmd
    assert "--vae" in cmd
    assert "--text_encoder" in cmd
    assert cmd[cmd.index("--text_encoder") + 1] == "C:/fake/qwen25vl"
    assert "--byt5" in cmd
    assert cmd[cmd.index("--byt5") + 1] == "C:/fake/byt5"
    # 1.5 single text-encoder design — no 1.0-style dual TE flags
    assert "--text_encoder1" not in cmd
    assert "--text_encoder2" not in cmd
    # task is passed (t2v default)
    assert "--task" in cmd
    assert cmd[cmd.index("--task") + 1] == "t2v"
    # steps / lr / seed
    assert "--max_train_steps" in cmd
    assert cmd[cmd.index("--max_train_steps") + 1] == "200"
    assert "--learning_rate" in cmd
    assert cmd[cmd.index("--learning_rate") + 1] == f"{config.learning_rate:.10g}"
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"
    # flow-matching flags
    assert "--timestep_sampling" in cmd
    assert "--discrete_flow_shift" in cmd
    assert spec.cwd == fake_musubi.resolve()
    assert "PYTHONIOENCODING" in spec.env
    assert spec.tfevents_glob.endswith("events.out.tfevents.*")
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()
    # sample dir is <output_dir>/sample
    assert spec.sample_dir is not None
    assert spec.sample_dir.name == "sample"
    assert spec.sample_dir.exists()


def test_prepare_run_i2v_emits_image_encoder(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = HunyuanVideo15LoRATrainer(
        musubi_dir=fake_musubi, venv_python=fake_python,
        dit_path="C:/fake/dit", vae_path="C:/fake/vae",
        text_encoder_path="C:/fake/qwen25vl", byt5_path="C:/fake/byt5",
        image_encoder_path="C:/fake/siglip", task="i2v", vram_gb=32.0,
    )
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=10, seed=0,
    )
    assert spec.cmd[spec.cmd.index("--task") + 1] == "i2v"
    assert "--image_encoder" in spec.cmd
    assert spec.cmd[spec.cmd.index("--image_encoder") + 1] == "C:/fake/siglip"


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


def test_prepare_run_rejects_wrong_config_type(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    from bracket.trainer.wan_lora import WanLoRAConfig

    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("", encoding="utf-8")
    with pytest.raises(TypeError):
        t.prepare_run(
            run_dir=tmp_path / "run", config=WanLoRAConfig(),
            dataset_toml=dataset_toml, max_steps=10, seed=0,
        )


def test_session_setup_commands_present_with_hv15_cache_modules(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    specs = t.session_setup_commands(
        dataset_toml=dataset_toml, run_dir=tmp_path / "run",
    )
    assert len(specs) == 2  # latents + text-encoder-outputs
    joined = [" ".join(s.cmd) for s in specs]
    # latent cache uses the hv_1_5 latent cache module + --vae
    assert any("hv_1_5_cache_latents" in j for j in joined)
    assert any("--vae" in j for j in joined)
    # TE cache uses the hv_1_5 TE cache module + --text_encoder + --byt5
    assert any("hv_1_5_cache_text_encoder_outputs" in j for j in joined)
    assert any("--text_encoder" in j and "--byt5" in j for j in joined)


def test_session_setup_commands_i2v_adds_image_encoder(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = HunyuanVideo15LoRATrainer(
        musubi_dir=fake_musubi, venv_python=fake_python,
        dit_path="C:/fake/dit", vae_path="C:/fake/vae",
        text_encoder_path="C:/fake/qwen25vl", byt5_path="C:/fake/byt5",
        image_encoder_path="C:/fake/siglip", task="i2v", vram_gb=32.0,
    )
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    specs = t.session_setup_commands(
        dataset_toml=dataset_toml, run_dir=tmp_path / "run",
    )
    joined = " ".join(" ".join(s.cmd) for s in specs)
    assert "--i2v" in joined
    assert "--image_encoder" in joined
