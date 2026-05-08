"""Verify that the LoRA trainer baselines pick fast defaults on high-VRAM
cards (gradient_checkpointing OFF, more dataloader workers) and safe
defaults on small-VRAM cards (checkpointing ON, fewer workers).

This is the user-visible bug we're guarding against: previously
`--gradient_checkpointing` was hardcoded ON which wastes ~25% step time on
a 32GB 5090 where it isn't needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.trainer.flux2_klein_lora import Flux2KleinLoRATrainer
from bracket.trainer.sdxl import SDXLTrainer
from bracket.trainer.zimage_lora import ZImageLoRATrainer


@pytest.fixture
def fake_python(tmp_path: Path) -> Path:
    p = tmp_path / "python.exe"; p.write_bytes(b""); return p


def _sd_scripts(tmp: Path) -> Path:
    d = tmp / "sd-scripts"; d.mkdir()
    (d / "sdxl_train_network.py").write_text("# fake\n", encoding="utf-8")
    return d


def _musubi(tmp: Path, scripts: list[str]) -> Path:
    d = tmp / "musubi"; d.mkdir(exist_ok=True)
    for s in scripts:
        (d / s).write_text("# fake\n", encoding="utf-8")
    return d


def _ds_toml(tmp: Path) -> Path:
    p = tmp / "ds.toml"
    p.write_text(
        "[general]\ncaption_extension=\".txt\"\n\n"
        "[[datasets]]\nresolution=[1024,1024]\nbatch_size=1\n"
        "  [[datasets.subsets]]\n  image_dir = \"C:/imgs\"\n  num_repeats=1\n",
        encoding="utf-8",
    )
    return p


def test_sdxl_high_vram_baseline_disables_grad_checkpointing(
    tmp_path: Path, fake_python: Path,
):
    """5090 / 32GB on the small SDXL (2.5B) model: baseline can safely
    disable ckpt — there's enough VRAM headroom."""
    t = SDXLTrainer(
        sd_scripts_dir=_sd_scripts(tmp_path), venv_python=fake_python,
        pretrained_model="C:/fake", vram_gb=32.0,
    )
    cfg = t.baseline_config()
    assert cfg.gradient_checkpointing is False
    assert cfg.dataloader_workers >= 6


def test_zimage_lora_keeps_ckpt_on_at_32gb(tmp_path: Path, fake_python: Path):
    """Z-Image is a 9B DiT — even on 32GB 5090, batch=4 without ckpt blows
    past VRAM. Baseline MUST keep ckpt=True or training paging hard."""
    t = ZImageLoRATrainer(
        musubi_dir=_musubi(tmp_path, ["zimage_train_network.py"]),
        venv_python=fake_python,
        dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    cfg = t.baseline_config()
    assert cfg.gradient_checkpointing is True
    # Musubi-tuner trainers keep workers low (2) per docs — multiple
    # [[datasets]] entries amplify worker count, and high `num_workers` with
    # `persistent_data_loader_workers` thrashes.
    assert cfg.dataloader_workers == 2


def test_flux2_klein_keeps_ckpt_on_at_32gb(tmp_path: Path, fake_python: Path):
    t = Flux2KleinLoRATrainer(
        musubi_dir=_musubi(tmp_path, ["flux_2_train_network.py"]),
        venv_python=fake_python,
        dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    cfg = t.baseline_config()
    assert cfg.gradient_checkpointing is True


def test_zimage_lora_can_consider_ckpt_off_at_xl_tier(tmp_path: Path, fake_python: Path):
    """80GB H100 has enough headroom for Z-Image batch=4 ckpt=off."""
    t = ZImageLoRATrainer(
        musubi_dir=_musubi(tmp_path, ["zimage_train_network.py"]),
        venv_python=fake_python,
        dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
        vram_gb=80.0,
    )
    from bracket.search.space import CategoricalKnob
    space = t.declare_search_space()
    knob = space.knobs["gradient_checkpointing"]
    assert isinstance(knob, CategoricalKnob)
    cfg = t.baseline_config()
    # On 80GB, even Z-Image baseline can skip ckpt
    assert cfg.gradient_checkpointing is False


def test_tiny_vram_keeps_grad_checkpointing_on(tmp_path: Path, fake_python: Path):
    """8GB card: keep ckpt=True so candidates fit. Knob is FixedKnob, not Categorical."""
    from bracket.search.space import CategoricalKnob, FixedKnob
    t = SDXLTrainer(
        sd_scripts_dir=_sd_scripts(tmp_path), venv_python=fake_python,
        pretrained_model="C:/fake", vram_gb=8.0,
    )
    space = t.declare_search_space()
    knob = space.knobs["gradient_checkpointing"]
    assert isinstance(knob, FixedKnob)
    assert knob.value is True
    cfg = t.baseline_config()
    assert cfg.gradient_checkpointing is True


def test_high_vram_sdxl_makes_grad_ckpt_a_search_knob(tmp_path: Path, fake_python: Path):
    """SDXL is small (2.5B) — on 32GB, orchestrator can search both ckpt
    options. Z-Image and Flux-2-Klein at 32GB cannot (covered above)."""
    from bracket.search.space import CategoricalKnob, FixedKnob
    sdxl = SDXLTrainer(
        sd_scripts_dir=_sd_scripts(tmp_path), venv_python=fake_python,
        pretrained_model="C:/fake", vram_gb=32.0,
    )
    sdxl_knob = sdxl.declare_search_space().knobs["gradient_checkpointing"]
    assert isinstance(sdxl_knob, CategoricalKnob)
    assert set(sdxl_knob.choices) == {True, False}

    zi = ZImageLoRATrainer(
        musubi_dir=_musubi(tmp_path, ["zimage_train_network.py"]),
        venv_python=fake_python,
        dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    zi_knob = zi.declare_search_space().knobs["gradient_checkpointing"]
    assert isinstance(zi_knob, FixedKnob), \
        "Z-Image 9B at 32GB cannot safely run ckpt=False — must be pinned True"
    assert zi_knob.value is True


def test_grad_ckpt_off_omits_cli_flag(tmp_path: Path, fake_python: Path):
    """When config says ckpt=False, --gradient_checkpointing must NOT be in cmd.
    SDXL on 32GB defaults ckpt=False so this exercises the omission path."""
    t = SDXLTrainer(
        sd_scripts_dir=_sd_scripts(tmp_path), venv_python=fake_python,
        pretrained_model="C:/fake", vram_gb=32.0,
    )
    cfg = t.baseline_config()
    assert cfg.gradient_checkpointing is False  # SDXL 2.5B fits without ckpt
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=cfg, dataset_toml=_ds_toml(tmp_path),
        max_steps=100, seed=0,
    )
    assert "--gradient_checkpointing" not in spec.cmd


def test_grad_ckpt_on_includes_cli_flag(tmp_path: Path, fake_python: Path):
    """When config says ckpt=True, --gradient_checkpointing IS in cmd."""
    from bracket.trainer.sdxl import SDXLConfig
    t = SDXLTrainer(
        sd_scripts_dir=_sd_scripts(tmp_path), venv_python=fake_python,
        pretrained_model="C:/fake", vram_gb=32.0,
    )
    cfg = t.baseline_config()
    cfg.gradient_checkpointing = True  # override
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=cfg, dataset_toml=_ds_toml(tmp_path),
        max_steps=100, seed=0,
    )
    assert "--gradient_checkpointing" in spec.cmd


def test_dataloader_workers_passed_through(tmp_path: Path, fake_python: Path):
    """The configured dataloader_workers count reaches --max_data_loader_n_workers."""
    t = ZImageLoRATrainer(
        musubi_dir=_musubi(tmp_path, ["zimage_train_network.py"]),
        venv_python=fake_python,
        dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    cfg = t.baseline_config()
    assert cfg.dataloader_workers == 2  # musubi pool — see hardware.py
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=cfg, dataset_toml=_ds_toml(tmp_path),
        max_steps=100, seed=0,
    )
    idx = spec.cmd.index("--max_data_loader_n_workers")
    assert spec.cmd[idx + 1] == "2"


def test_subprocess_env_caps_omp_threads(tmp_path: Path, fake_python: Path):
    """Trainer subprocess env must cap OMP/MKL threads — without this, every
    DataLoader worker spawns one OMP thread per CPU core, causing massive
    context switching and ~1000x slowdown on Blackwell GPUs (sm_120)."""
    t = ZImageLoRATrainer(
        musubi_dir=_musubi(tmp_path, ["zimage_train_network.py"]),
        venv_python=fake_python,
        dit_path="C:/d", vae_path="C:/v", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    cfg = t.baseline_config()
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=cfg, dataset_toml=_ds_toml(tmp_path),
        max_steps=100, seed=0,
    )
    assert spec.env.get("OMP_NUM_THREADS") == "1"
    assert spec.env.get("MKL_NUM_THREADS") == "1"
    assert spec.env.get("OPENBLAS_NUM_THREADS") == "1"
