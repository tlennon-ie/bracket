"""Tests for the Krea2LoRATrainer adapter.

We never launch musubi-tuner — just verify cmdline construction (the
``krea2_train_network.py`` script + ``--network_module networks.lora_krea2``,
the Krea-2 weight trio, and the flow-matching timestep flags), search-space
shape, config roundtripping, and the two krea2 pre-cache commands.

Background: Krea 2 (K2) is a single-stream MMDiT text-to-image model. It uses
the **Qwen-Image VAE** (same as the Qwen-Image integration) and a single
**Qwen3-VL-4B-Instruct** text encoder. The LoRA workflow trains on the RAW DiT
with ``--timestep_sampling shift --discrete_flow_shift 2.5`` (2.5 matches the
K2 1024px inference time-shift). See vendor/musubi-tuner/docs/krea2.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import CategoricalKnob, FixedKnob, FloatKnob
from bracket.trainer.krea2_lora import Krea2LoRAConfig, Krea2LoRATrainer


@pytest.fixture
def fake_musubi(tmp_path: Path) -> Path:
    """Pretend musubi-tuner dir with the krea2 train script."""
    d = tmp_path / "musubi-tuner"
    pkg = d / "src" / "musubi_tuner"
    pkg.mkdir(parents=True)
    (pkg / "krea2_train_network.py").write_text("# placeholder\n", encoding="utf-8")
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
    vram_gb: float = 32.0,
) -> Krea2LoRATrainer:
    return Krea2LoRATrainer(
        musubi_dir=fake_musubi,
        venv_python=fake_python,
        dit_path="C:/fake/krea2_raw.safetensors",
        vae_path="C:/fake/qwen_image_vae.safetensors",
        text_encoder_path="C:/fake/qwen3vl_4b_bf16.safetensors",
        vram_gb=vram_gb,
    )


def test_init_uses_krea2_train_script(fake_musubi: Path, fake_python: Path):
    t = _make_trainer(fake_musubi, fake_python)
    assert t.train_script.name == "krea2_train_network.py"
    assert t.train_script.exists()


def test_blocks_to_swap_capped_at_dit_max(fake_musubi: Path, fake_python: Path):
    # Krea 2 has 28 main blocks; musubi caps --blocks_to_swap at 26. Even the
    # low/med tiers (table values 36/28) must be clamped.
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        assert t.baseline_config().blocks_to_swap <= 26
        assert t.declare_search_space().knobs["blocks_to_swap"].value <= 26


def test_init_rejects_missing_musubi(tmp_path: Path, fake_python: Path):
    with pytest.raises(FileNotFoundError):
        Krea2LoRATrainer(
            musubi_dir=tmp_path / "nope",
            venv_python=fake_python,
            dit_path="C:/fake/dit",
            vae_path="C:/fake/vae",
            text_encoder_path="C:/fake/te",
        )


def test_init_rejects_missing_python(fake_musubi: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Krea2LoRATrainer(
            musubi_dir=fake_musubi,
            venv_python=tmp_path / "no_python.exe",
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
    assert "AdamW8bit" in space.knobs["optimizer_type"].choices
    assert isinstance(space.knobs["lr_scheduler"], CategoricalKnob)
    assert isinstance(space.knobs["network_dim"], CategoricalKnob)
    assert isinstance(space.knobs["network_alpha"], CategoricalKnob)
    # Flow-matching shift is searchable, centered ~2.5.
    assert isinstance(space.knobs["discrete_flow_shift"], FloatKnob)
    assert space.knobs["discrete_flow_shift"].low <= 2.5 <= space.knobs["discrete_flow_shift"].high
    # Non-loss-bearing knobs are pinned.
    assert isinstance(space.knobs["fp8_base"], FixedKnob)
    assert isinstance(space.knobs["fp8_scaled"], FixedKnob)
    assert isinstance(space.knobs["blocks_to_swap"], FixedKnob)
    assert isinstance(space.knobs["mixed_precision"], FixedKnob)


def test_baseline_config_validates_against_search_space(fake_musubi: Path, fake_python: Path):
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


def test_curated_configs_validate_against_search_space(fake_musubi: Path, fake_python: Path):
    """Curated configs must validate against the search space — EXCEPT the
    learning_rate (curated configs intentionally use out-of-band LR like
    Prodigy's 1.0; the orchestrator clamps LR at runtime)."""
    for vram in (8.0, 24.0, 32.0, 80.0):
        t = _make_trainer(fake_musubi, fake_python, vram_gb=vram)
        space = t.declare_search_space()
        for cfg in t.curated_configs():
            assert isinstance(cfg, Krea2LoRAConfig)
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
        "discrete_flow_shift": 3.2, "train_batch_size": 1,
        "gradient_accumulation_steps": 1, "mixed_precision": "bf16",
        "max_grad_norm": 1.0, "fp8_base": True, "fp8_scaled": True,
        "gradient_checkpointing": False, "blocks_to_swap": 0,
        "dataloader_workers": 2,
    }
    config = t.config_from_dict(knobs)
    assert isinstance(config, Krea2LoRAConfig)
    assert config.learning_rate == 5e-5
    assert config.optimizer_type == "Lion"
    assert config.network_dim == 16
    assert config.discrete_flow_shift == 3.2


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
    # Wrapped via `accelerate launch`.
    assert cmd[0] == str(fake_python.resolve())
    assert "accelerate.commands.launch" in cmd
    assert "--num_cpu_threads_per_process" in cmd
    assert any("krea2_train_network.py" in part for part in cmd)
    # Krea-2 weight trio.
    assert "--dit" in cmd
    assert cmd[cmd.index("--dit") + 1] == "C:/fake/krea2_raw.safetensors"
    assert "--vae" in cmd
    assert cmd[cmd.index("--vae") + 1] == "C:/fake/qwen_image_vae.safetensors"
    assert "--text_encoder" in cmd
    assert cmd[cmd.index("--text_encoder") + 1] == "C:/fake/qwen3vl_4b_bf16.safetensors"
    # Required Krea-2 LoRA network module.
    assert "--network_module" in cmd
    assert cmd[cmd.index("--network_module") + 1] == "networks.lora_krea2"
    # Flow-matching timestep flags (documented starting point).
    assert "--timestep_sampling" in cmd
    assert cmd[cmd.index("--timestep_sampling") + 1] == "shift"
    assert "--discrete_flow_shift" in cmd
    assert "--max_train_steps" in cmd
    assert cmd[cmd.index("--max_train_steps") + 1] == "200"
    assert "--learning_rate" in cmd
    assert cmd[cmd.index("--learning_rate") + 1] == f"{config.learning_rate:.10g}"
    # Inference-only flags must NEVER appear in the TRAIN command.
    assert "--guidance_scale" not in cmd
    assert "--steps" not in cmd
    assert "--turbo_dit" not in cmd
    assert spec.cwd == fake_musubi.resolve()
    assert "PYTHONIOENCODING" in spec.env
    assert spec.tfevents_glob.endswith("events.out.tfevents.*")
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()
    # Sample dir is <output>/sample (musubi default — singular).
    assert spec.sample_dir is not None
    assert spec.sample_dir.name == "sample"
    assert spec.sample_dir.exists()


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


def test_session_setup_emits_both_krea2_cache_commands(
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
    # Krea-2-specific cache modules.
    assert "musubi_tuner.krea2_cache_latents" in latents_cmd
    assert "musubi_tuner.krea2_cache_text_encoder_outputs" in te_cmd
    # Latent cache needs the (Qwen-Image) VAE; TE cache needs the Qwen3-VL TE.
    assert "--vae" in latents_cmd
    assert latents_cmd[latents_cmd.index("--vae") + 1] == "C:/fake/qwen_image_vae.safetensors"
    assert "--text_encoder" in te_cmd
    assert te_cmd[te_cmd.index("--text_encoder") + 1] == "C:/fake/qwen3vl_4b_bf16.safetensors"
    # Pre-cache steps produce no tfevents.
    assert specs[0].tfevents_glob == ""
    assert specs[1].tfevents_glob == ""
