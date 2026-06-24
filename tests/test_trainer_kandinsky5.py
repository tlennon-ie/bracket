"""Tests for the Kandinsky5LoRATrainer adapter. We never launch musubi-tuner —
just verify cmdline construction, search-space declaration, config roundtripping,
and the dual-text-encoder pre-cache commands."""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FixedKnob, FloatKnob, IntKnob
from bracket.trainer.kandinsky5_lora import (
    Kandinsky5LoRAConfig,
    Kandinsky5LoRATrainer,
)


@pytest.fixture
def fake_musubi(tmp_path: Path) -> Path:
    """Pretend musubi-tuner dir with the expected nested train script."""
    d = tmp_path / "musubi-tuner"
    pkg = d / "src" / "musubi_tuner"
    pkg.mkdir(parents=True)
    (pkg / "kandinsky5_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    return d


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def _make_trainer(fake_musubi: Path, fake_python: Path, vram_gb: float = 32.0):
    return Kandinsky5LoRATrainer(
        musubi_dir=fake_musubi,
        venv_python=fake_python,
        dit_path="C:/fake/kandinsky5_dit.safetensors",
        vae_path="C:/fake/kandinsky5_vae.safetensors",
        text_encoder_qwen_path="C:/fake/qwen2.5-vl",
        text_encoder_clip_path="C:/fake/clip-vit-large",
        vram_gb=vram_gb,
    )


def test_init_rejects_missing_musubi(tmp_path: Path, fake_python: Path):
    with pytest.raises(FileNotFoundError):
        Kandinsky5LoRATrainer(
            musubi_dir=tmp_path / "nope",
            venv_python=fake_python,
            dit_path="C:/fake/dit",
            vae_path="C:/fake/vae",
            text_encoder_qwen_path="C:/fake/qwen",
            text_encoder_clip_path="C:/fake/clip",
        )


def test_init_rejects_missing_python(fake_musubi: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Kandinsky5LoRATrainer(
            musubi_dir=fake_musubi,
            venv_python=tmp_path / "no_python.exe",
            dit_path="C:/fake/dit",
            vae_path="C:/fake/vae",
            text_encoder_qwen_path="C:/fake/qwen",
            text_encoder_clip_path="C:/fake/clip",
        )


def test_search_space_has_expected_knobs(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    space = t.declare_search_space()
    assert "learning_rate" in space.knobs
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    assert isinstance(space.knobs["optimizer_type"], CategoricalKnob)
    assert "AdamW8bit" in space.knobs["optimizer_type"].choices
    assert isinstance(space.knobs["lr_warmup_steps"], IntKnob)
    assert isinstance(space.knobs["discrete_flow_shift"], FloatKnob)
    # Non-loss-bearing knobs are pinned.
    assert isinstance(space.knobs["fp8_base"], FixedKnob)
    assert isinstance(space.knobs["blocks_to_swap"], FixedKnob)


def test_search_space_batch_size_scales_with_vram(fake_musubi: Path, fake_python: Path):
    t_low = _make_trainer(fake_musubi, fake_python, vram_gb=8.0)
    t_high = _make_trainer(fake_musubi, fake_python, vram_gb=32.0)
    low_choices = t_low.declare_search_space().knobs["train_batch_size"].choices
    high_choices = t_high.declare_search_space().knobs["train_batch_size"].choices
    assert max(high_choices) >= max(low_choices)
    assert max(low_choices) == 1  # tiny tier pins to 1


def test_baseline_config_validates_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


def test_curated_configs_validate_against_search_space(fake_musubi: Path, fake_python: Path):
    # learning_rate is exempt: Prodigy curated configs intentionally use LR=1.0
    # (Prodigy auto-tunes), which the orchestrator clamps via
    # clamp_config_to_overrides rather than the static knob range — mirrors the
    # qwen/zimage/flux curated configs.
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        curated = t.curated_configs()
        assert len(curated) >= 1
        for cfg in curated:
            d = cfg.to_dict()
            for k, knob in space.knobs.items():
                if k in d and k != "learning_rate":
                    knob.validate(d[k])


def test_config_from_dict_roundtrips(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    knobs = {
        "learning_rate": 5e-5, "optimizer_type": "Lion",
        "lr_scheduler": "constant", "lr_warmup_steps": 50,
        "network_dim": 16, "network_alpha": 8.0,
        "discrete_flow_shift": 6.0, "train_batch_size": 1,
        "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
        "max_grad_norm": 1.0, "fp8_base": True,
        "gradient_checkpointing": False, "blocks_to_swap": 0,
        "dataloader_workers": 2,
    }
    config = t.config_from_dict(knobs)
    assert config.learning_rate == 5e-5
    assert config.optimizer_type == "Lion"
    assert config.network_dim == 16
    assert config.discrete_flow_shift == 6.0


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
    # Wrapped via `accelerate launch`.
    assert "accelerate.commands.launch" in cmd
    assert "--num_cpu_threads_per_process" in cmd
    assert any("kandinsky5_train_network.py" in part for part in cmd)
    assert "--max_train_steps" in cmd
    assert cmd[cmd.index("--max_train_steps") + 1] == "200"
    assert "--learning_rate" in cmd
    assert cmd[cmd.index("--learning_rate") + 1] == f"{config.learning_rate:.10g}"
    # Kandinsky-specific network module.
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora_kandinsky"
    # Both text encoders present, each with its own flag + path.
    assert "--text_encoder_qwen" in cmd
    assert cmd[cmd.index("--text_encoder_qwen") + 1] == "C:/fake/qwen2.5-vl"
    assert "--text_encoder_clip" in cmd
    assert cmd[cmd.index("--text_encoder_clip") + 1] == "C:/fake/clip-vit-large"
    # DiT + VAE weights.
    assert "--dit" in cmd
    assert cmd[cmd.index("--dit") + 1] == "C:/fake/kandinsky5_dit.safetensors"
    assert "--vae" in cmd
    # Task selector + flow-matching flags.
    assert "--task" in cmd
    assert cmd[cmd.index("--task") + 1] == "k5-pro-t2v-5s-sd"
    assert "--timestep_sampling" in cmd
    assert cmd[cmd.index("--timestep_sampling") + 1] == "shift"
    assert "--discrete_flow_shift" in cmd
    # Runs from the musubi dir; thread-cap env present.
    assert spec.cwd == fake_musubi.resolve()
    assert "PYTHONIOENCODING" in spec.env
    assert spec.tfevents_glob.endswith("events.out.tfevents.*")
    # Sample dir is <output_dir>/sample.
    assert spec.sample_dir == spec.output_dir / "sample"
    # Output directories were created.
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()
    assert spec.sample_dir.exists()


def test_prepare_run_omits_optional_args_when_none(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    config = t.baseline_config()
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=config, dataset_toml=dataset_toml,
        max_steps=10, seed=0,
    )
    # No sampling, no resume, no save-state by default.
    assert "--sample_prompts" not in spec.cmd
    assert "--sample_every_n_steps" not in spec.cmd
    assert "--resume" not in spec.cmd
    assert "--save_state" not in spec.cmd


def test_prepare_run_rejects_wrong_config_type(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")

    class _Wrong(Kandinsky5LoRAConfig):
        pass

    # A genuinely different type must be rejected.
    with pytest.raises(TypeError):
        t.prepare_run(
            run_dir=tmp_path / "run", config=object(),  # type: ignore[arg-type]
            dataset_toml=dataset_toml, max_steps=10, seed=0,
        )


def test_session_setup_emits_dual_te_precache(
    fake_musubi: Path, fake_python: Path, tmp_path: Path,
):
    t = _make_trainer(fake_musubi, fake_python)
    dataset_toml = tmp_path / "ds.toml"
    dataset_toml.write_text("[general]\n", encoding="utf-8")
    specs = t.session_setup_commands(
        dataset_toml=dataset_toml, run_dir=tmp_path / "run",
    )
    assert len(specs) == 2
    latents_cmd, te_cmd = specs[0].cmd, specs[1].cmd
    # Cache-latents command.
    assert "musubi_tuner.kandinsky5_cache_latents" in latents_cmd
    assert "--vae" in latents_cmd
    # Cache-text-encoder-outputs command names BOTH encoders.
    assert "musubi_tuner.kandinsky5_cache_text_encoder_outputs" in te_cmd
    assert "--text_encoder_qwen" in te_cmd
    assert te_cmd[te_cmd.index("--text_encoder_qwen") + 1] == "C:/fake/qwen2.5-vl"
    assert "--text_encoder_clip" in te_cmd
    assert te_cmd[te_cmd.index("--text_encoder_clip") + 1] == "C:/fake/clip-vit-large"
    # Pre-cache specs carry no tfevents glob.
    assert specs[0].tfevents_glob == ""
    assert specs[1].tfevents_glob == ""
