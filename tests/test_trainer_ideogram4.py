"""Tests for the Ideogram4LoRATrainer adapter (musubi-tuner ideogram4_train_network).

We never launch the real trainer — these tests verify search-space declaration,
config roundtripping, curated/baseline validation, and cmdline construction with
a fake musubi_dir + fake train script (the SDXL / qwen-image test pattern).

Ideogram 4 (musubi docs/ideogram4.md) is a single-TE image LoRA adapter:
  - --dit loads an FP8-only frozen base (no --fp8_base flag); --dit_dtype is the
    compute dtype (bfloat16 default)
  - --vae is the Flux2 KL-VAE (--vae_dtype bfloat16); --text_encoder is Qwen3-VL-8B
  - flow-matching uses --timestep_sampling ideogram4_shift + --weighting_scheme none
  - LoRA network module is networks.lora_ideogram4 (required)
  - pre-cache scripts: ideogram4_cache_latents (needs --vae) and
    ideogram4_cache_text_encoder_outputs (needs --text_encoder)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FloatKnob
from bracket.trainer.ideogram4_lora import Ideogram4LoRAConfig, Ideogram4LoRATrainer


@pytest.fixture
def fake_musubi(tmp_path: Path) -> Path:
    """Pretend musubi-tuner dir with the Ideogram 4 LoRA train script in place."""
    d = tmp_path / "musubi-tuner"
    pkg = d / "src" / "musubi_tuner"
    pkg.mkdir(parents=True)
    (pkg / "ideogram4_train_network.py").write_text("# placeholder\n", encoding="utf-8")
    return d


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def _make_trainer(
    fake_musubi: Path, fake_python: Path, vram_gb: float = 32.0,
) -> Ideogram4LoRATrainer:
    return Ideogram4LoRATrainer(
        musubi_dir=fake_musubi,
        venv_python=fake_python,
        dit_path="C:/fake/ideogram4_fp8_scaled.safetensors",
        vae_path="C:/fake/flux2-vae.safetensors",
        text_encoder_path="C:/fake/qwen3vl_8b_fp8_scaled.safetensors",
        vram_gb=vram_gb,
    )


def test_init_rejects_missing_musubi(tmp_path: Path, fake_python: Path):
    with pytest.raises(FileNotFoundError):
        Ideogram4LoRATrainer(
            musubi_dir=tmp_path / "nope",
            venv_python=fake_python,
            dit_path="C:/fake/dit.safetensors",
            vae_path="C:/fake/vae.safetensors",
            text_encoder_path="C:/fake/te.safetensors",
        )


def test_init_rejects_missing_python(fake_musubi: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Ideogram4LoRATrainer(
            musubi_dir=fake_musubi,
            venv_python=tmp_path / "no_python.exe",
            dit_path="C:/fake/dit.safetensors",
            vae_path="C:/fake/vae.safetensors",
            text_encoder_path="C:/fake/te.safetensors",
        )


def test_name_is_ideogram4_lora(fake_musubi: Path, fake_python: Path):
    assert _make_trainer(fake_musubi, fake_python).name == "ideogram4-lora"


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


def test_blocks_to_swap_capped_at_dit_max(fake_musubi: Path, fake_python: Path):
    # Ideogram 4 DiT is 34 layers; musubi caps --blocks_to_swap at 33. Even the
    # tiniest tier (default table value 40) must be clamped.
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        assert t.baseline_config().blocks_to_swap <= 33


def test_baseline_config_validates_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


def test_curated_configs_validate_against_search_space(fake_musubi: Path, fake_python: Path):
    # The orchestrator clamps LR at runtime, so curated LRs are exempt from the
    # FloatKnob LR range; every other knob must validate against the space.
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        curated = t.curated_configs()
        assert len(curated) >= 1
        for config in curated:
            d = config.to_dict()
            for k, knob in space.knobs.items():
                if k == "learning_rate":
                    continue  # runtime-clamped; curated LR exemption
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
    assert any("ideogram4_train_network.py" in part for part in cmd)
    # Ideogram 4 LoRA network module is required
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora_ideogram4"
    # single-TE image weights: --dit, --vae, --text_encoder
    assert "--dit" in cmd
    assert cmd[cmd.index("--dit") + 1] == "C:/fake/ideogram4_fp8_scaled.safetensors"
    assert "--vae" in cmd
    assert cmd[cmd.index("--vae") + 1] == "C:/fake/flux2-vae.safetensors"
    assert "--text_encoder" in cmd
    assert cmd[cmd.index("--text_encoder") + 1] == "C:/fake/qwen3vl_8b_fp8_scaled.safetensors"
    # dtype flags: FP8 base is implicit, so no --fp8_base; compute dtypes set
    assert "--fp8_base" not in cmd
    assert "--dit_dtype" in cmd
    assert cmd[cmd.index("--dit_dtype") + 1] == "bfloat16"
    assert "--vae_dtype" in cmd
    assert cmd[cmd.index("--vae_dtype") + 1] == "bfloat16"
    # Ideogram-4-specific flow-matching flags
    assert "--timestep_sampling" in cmd
    assert cmd[cmd.index("--timestep_sampling") + 1] == "ideogram4_shift"
    assert "--weighting_scheme" in cmd
    assert cmd[cmd.index("--weighting_scheme") + 1] == "none"
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


def test_session_setup_commands_use_ideogram4_cache_scripts(
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
    latents_cmd = " ".join(specs[0].cmd)
    te_cmd = " ".join(specs[1].cmd)
    assert "ideogram4_cache_latents" in latents_cmd
    assert "ideogram4_cache_text_encoder_outputs" in te_cmd
    # latent cache passes --vae; TE cache passes --text_encoder
    assert "--vae" in specs[0].cmd
    assert specs[0].cmd[specs[0].cmd.index("--vae") + 1] == "C:/fake/flux2-vae.safetensors"
    assert "--text_encoder" in specs[1].cmd
    assert (
        specs[1].cmd[specs[1].cmd.index("--text_encoder") + 1]
        == "C:/fake/qwen3vl_8b_fp8_scaled.safetensors"
    )
    for spec in specs:
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
