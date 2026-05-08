"""Tests for the new trainer adapters: SDXL full-FT, Z-Image LoRA, Z-Image full-FT.

Same test pattern as test_trainer_sdxl.py — fake script files + fake python so
we never actually launch a trainer. We verify the cmdlines, search spaces,
and config roundtripping."""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.search.space import (
    CategoricalKnob,
    FloatKnob,
    IntKnob,
)
from bracket.trainer.sdxl_full import SDXLFullTrainer, SDXLFullConfig
from bracket.trainer.zimage_full import ZImageFullTrainer, ZImageFullConfig
from bracket.trainer.zimage_lora import ZImageLoRATrainer, ZImageLoRAConfig


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"
    p.write_bytes(b"")
    return p


def _make_sd_scripts_with(tmp: Path, scripts: list[str]) -> Path:
    d = tmp / "sd-scripts"
    d.mkdir(exist_ok=True)
    for s in scripts:
        (d / s).write_text("# fake\n", encoding="utf-8")
    return d


def _make_musubi_with(tmp: Path, scripts: list[str], top_level: bool = True) -> Path:
    d = tmp / "musubi"
    d.mkdir(exist_ok=True)
    if top_level:
        for s in scripts:
            (d / s).write_text("# fake\n", encoding="utf-8")
    else:
        sd = d / "src" / "musubi_tuner"
        sd.mkdir(parents=True, exist_ok=True)
        for s in scripts:
            (sd / s).write_text("# fake\n", encoding="utf-8")
    return d


# ───────── SDXL full-FT ─────────


def test_sdxl_full_search_space_excludes_lora_knobs(tmp_path: Path, fake_python: Path):
    sd = _make_sd_scripts_with(tmp_path, ["sdxl_train.py"])
    t = SDXLFullTrainer(sd_scripts_dir=sd, venv_python=fake_python,
                        pretrained_model="C:/fake", vram_gb=32.0)
    knobs = t.declare_search_space().knobs
    # Full-FT specifically does NOT search rank/alpha
    assert "network_dim" not in knobs
    assert "network_alpha" not in knobs
    # But does search lr range appropriate for full-FT
    lr = knobs["learning_rate"]
    assert isinstance(lr, FloatKnob) and lr.log
    assert lr.high <= 1e-5  # full-FT range, not LoRA's 5e-4


def test_sdxl_full_baseline_uses_adafactor_and_fused_backward(tmp_path: Path, fake_python: Path):
    sd = _make_sd_scripts_with(tmp_path, ["sdxl_train.py"])
    t = SDXLFullTrainer(sd_scripts_dir=sd, venv_python=fake_python,
                        pretrained_model="C:/fake", vram_gb=32.0)
    cfg = t.baseline_config()
    assert cfg.optimizer_type == "Adafactor"
    assert cfg.fused_backward_pass is True
    assert cfg.max_grad_norm == 0.0  # required when using fused backward


def test_sdxl_full_prepare_run_emits_correct_script_and_optimizer_args(
    tmp_path: Path, fake_python: Path,
):
    sd = _make_sd_scripts_with(tmp_path, ["sdxl_train.py"])
    t = SDXLFullTrainer(sd_scripts_dir=sd, venv_python=fake_python,
                        pretrained_model="C:/fake", vram_gb=32.0)
    cfg = t.baseline_config()
    ds = tmp_path / "ds.toml"; ds.write_text("[general]", encoding="utf-8")
    spec = t.prepare_run(run_dir=tmp_path / "run", config=cfg, dataset_toml=ds,
                        max_steps=100, seed=7)
    assert "sdxl_train.py" in spec.cmd
    assert "accelerate.commands.launch" in spec.cmd
    # sdxl_train.py (full-FT) doesn't accept --no_metadata; that's a LoRA-only
    # flag in sd-scripts. Including it crashes the trainer at argparse.
    assert "--no_metadata" not in spec.cmd
    # Must NOT include LoRA-specific flags
    assert "--network_module" not in spec.cmd
    assert "--network_dim" not in spec.cmd
    # Must include full-FT defining flags
    assert "--full_bf16" in spec.cmd
    assert "--fused_backward_pass" in spec.cmd
    # Adafactor optimizer args present
    assert "--optimizer_args" in spec.cmd
    args_idx = spec.cmd.index("--optimizer_args")
    # The next 3 strings should be the relative_step/scale_parameter/warmup_init flags
    next_three = spec.cmd[args_idx + 1: args_idx + 4]
    assert "relative_step=False" in next_three
    assert "scale_parameter=False" in next_three


def test_sdxl_full_baseline_validates_against_search_space(tmp_path: Path, fake_python: Path):
    sd = _make_sd_scripts_with(tmp_path, ["sdxl_train.py"])
    for vram in (8.0, 16.0, 24.0, 32.0, 48.0, 80.0):
        t = SDXLFullTrainer(sd_scripts_dir=sd, venv_python=fake_python,
                            pretrained_model="C:/fake", vram_gb=vram)
        space = t.declare_search_space()
        baseline = t.baseline_config().to_dict()
        for k, knob in space.knobs.items():
            if k in baseline:
                knob.validate(baseline[k])


# ───────── Z-Image LoRA ─────────


def test_zimage_lora_uses_top_level_script_when_present(tmp_path: Path, fake_python: Path):
    m = _make_musubi_with(tmp_path, ["zimage_train_network.py"], top_level=True)
    t = ZImageLoRATrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    assert t.train_script.name == "zimage_train_network.py"


def test_zimage_lora_uses_src_layout_when_top_level_missing(tmp_path: Path, fake_python: Path):
    m = _make_musubi_with(tmp_path, ["zimage_train_network.py"], top_level=False)
    t = ZImageLoRATrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    assert "src" in t.train_script.parts
    assert t.train_script.name == "zimage_train_network.py"


def test_zimage_lora_search_space_includes_discrete_flow_shift(tmp_path: Path, fake_python: Path):
    m = _make_musubi_with(tmp_path, ["zimage_train_network.py"])
    t = ZImageLoRATrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    knobs = t.declare_search_space().knobs
    assert "discrete_flow_shift" in knobs
    assert "network_dim" in knobs


def test_zimage_lora_prepare_run_uses_lora_zimage_module(tmp_path: Path, fake_python: Path):
    m = _make_musubi_with(tmp_path, ["zimage_train_network.py"])
    t = ZImageLoRATrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    cfg = t.baseline_config()
    ds = tmp_path / "ds.toml"
    ds.write_text(
        "[general]\ncaption_extension=\".txt\"\n\n"
        "[[datasets]]\nresolution=[1024,1024]\nbatch_size=1\n"
        "  [[datasets.subsets]]\n  image_dir = \"C:/imgs/a\"\n  num_repeats = 4\n",
        encoding="utf-8",
    )
    spec = t.prepare_run(run_dir=tmp_path / "run", config=cfg, dataset_toml=ds,
                        max_steps=100, seed=42)
    assert "--network_module" in spec.cmd
    assert spec.cmd[spec.cmd.index("--network_module") + 1] == "networks.lora_zimage"
    assert "--dit" in spec.cmd
    assert "--vae" in spec.cmd
    assert "--text_encoder" in spec.cmd
    assert "--discrete_flow_shift" in spec.cmd
    # musubi-tuner does NOT accept these — we pass them via dataset TOML or skip
    assert "--train_batch_size" not in spec.cmd
    assert "--noise_offset" not in spec.cmd
    assert "--min_snr_gamma" not in spec.cmd
    assert "--cache_latents" not in spec.cmd
    # Per-run TOML was written in musubi format (flat) with batch_size in [general]
    import toml as _toml
    run_toml = _toml.load(spec.cmd[spec.cmd.index("--dataset_config") + 1])
    assert run_toml["general"]["batch_size"] == cfg.train_batch_size
    # And no nested 'subsets' (would be rejected by musubi schema)
    for ds in run_toml["datasets"]:
        assert "subsets" not in ds


# ───────── Z-Image full-FT ─────────


def test_zimage_full_baseline_matches_user_production_settings(tmp_path: Path, fake_python: Path):
    m = _make_musubi_with(tmp_path, ["zimage_train.py"])
    t = ZImageFullTrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    cfg = t.baseline_config()
    # The user's actual prod run used Adafactor + lr=1e-6 + warmup=10
    assert cfg.optimizer_type == "Adafactor"
    assert cfg.learning_rate == 1e-6
    assert cfg.lr_warmup_steps == 10
    assert cfg.fused_backward_pass is True
    assert cfg.max_grad_norm == 0.0


def test_zimage_full_curated_configs_validate_against_search_space(tmp_path: Path, fake_python: Path):
    """Every curated config must be expressible in the trainer's search
    space — otherwise the orchestrator would crash trying to record it."""
    m = _make_musubi_with(tmp_path, ["zimage_train.py"])
    t = ZImageFullTrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    space = t.declare_search_space()
    curated = t.curated_configs()
    assert len(curated) >= 3, "z-image full-FT should ship multiple curated configs"
    for cfg in curated:
        knobs = cfg.to_dict()
        # Drop fields not in the search space (these are config-only, not knobs)
        knobs = {k: v for k, v in knobs.items() if k in space.knobs}
        space.validate(knobs)  # raises if any knob is out of declared range


def test_zimage_full_search_space_excludes_adamw(tmp_path: Path, fake_python: Path):
    """musubi's zimage_train.py --fused_backward_pass crashes on AdamW with
    KeyError('beta1'). The optimizer knob must offer Adafactor only."""
    m = _make_musubi_with(tmp_path, ["zimage_train.py"])
    t = ZImageFullTrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    knobs = t.declare_search_space().knobs
    opt = knobs["optimizer_type"]
    assert tuple(opt.choices) == ("Adafactor",)


def test_zimage_full_prepare_run_no_lora_module(tmp_path: Path, fake_python: Path):
    m = _make_musubi_with(tmp_path, ["zimage_train.py"])
    t = ZImageFullTrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    cfg = t.baseline_config()
    ds = tmp_path / "ds.toml"
    ds.write_text(
        "[general]\ncaption_extension=\".txt\"\n\n[[datasets]]\nresolution=[1024,1024]\n",
        encoding="utf-8",
    )
    spec = t.prepare_run(run_dir=tmp_path / "run", config=cfg, dataset_toml=ds,
                        max_steps=100, seed=42)
    assert "--network_module" not in spec.cmd
    assert "--network_dim" not in spec.cmd
    assert "--full_bf16" in spec.cmd
    assert "--fused_backward_pass" in spec.cmd
    assert "--mem_eff_save" in spec.cmd  # zimage full-FT writes large checkpoints
    # musubi-tuner has no --train_batch_size flag
    assert "--train_batch_size" not in spec.cmd


def test_zimage_full_blocks_to_swap_only_emitted_when_positive(tmp_path: Path, fake_python: Path):
    m = _make_musubi_with(tmp_path, ["zimage_train.py"])
    t = ZImageFullTrainer(musubi_dir=m, venv_python=fake_python,
                          dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
                          vram_gb=32.0)
    cfg = t.baseline_config()  # blocks_to_swap=0
    ds = tmp_path / "ds.toml"
    ds.write_text(
        "[general]\ncaption_extension=\".txt\"\n\n[[datasets]]\nresolution=[1024,1024]\n",
        encoding="utf-8",
    )
    spec = t.prepare_run(run_dir=tmp_path / "run", config=cfg, dataset_toml=ds,
                        max_steps=10, seed=0)
    assert "--blocks_to_swap" not in spec.cmd
    cfg2 = ZImageFullConfig(blocks_to_swap=16)
    spec2 = t.prepare_run(run_dir=tmp_path / "run2", config=cfg2, dataset_toml=ds,
                         max_steps=10, seed=0)
    assert "--blocks_to_swap" in spec2.cmd
    assert spec2.cmd[spec2.cmd.index("--blocks_to_swap") + 1] == "16"


# ───────── Cross-trainer sanity ─────────


def test_all_trainers_have_distinct_search_space_names(tmp_path: Path, fake_python: Path):
    """Different trainer adapters should produce search spaces with different
    names so the orchestrator can detect cross-trainer comparisons."""
    sd = _make_sd_scripts_with(tmp_path, ["sdxl_train.py", "sdxl_train_network.py"])
    m = _make_musubi_with(tmp_path, ["zimage_train.py", "zimage_train_network.py"])
    from bracket.trainer.sdxl import SDXLTrainer
    sdxl_lora = SDXLTrainer(sd_scripts_dir=sd, venv_python=fake_python, vram_gb=32.0)
    sdxl_full = SDXLFullTrainer(sd_scripts_dir=sd, venv_python=fake_python,
                                pretrained_model="C:/fake", vram_gb=32.0)
    zi_lora = ZImageLoRATrainer(musubi_dir=m, venv_python=fake_python,
                                dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
                                vram_gb=32.0)
    zi_full = ZImageFullTrainer(musubi_dir=m, venv_python=fake_python,
                                dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
                                vram_gb=32.0)
    names = {
        sdxl_lora.declare_search_space().name,
        sdxl_full.declare_search_space().name,
        zi_lora.declare_search_space().name,
        zi_full.declare_search_space().name,
    }
    assert len(names) == 4
