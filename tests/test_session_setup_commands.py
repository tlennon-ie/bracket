"""Tests for trainer.session_setup_commands and orchestrator's invocation
of them once-per-session."""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.trainer.flux2_klein_lora import Flux2KleinLoRATrainer
from bracket.trainer.sdxl import SDXLTrainer
from bracket.trainer.zimage_full import ZImageFullTrainer
from bracket.trainer.zimage_lora import ZImageLoRATrainer


def _python(tmp: Path) -> Path:
    p = tmp / "python.exe"; p.write_bytes(b""); return p


def _write_subset_toml(tmp: Path) -> Path:
    """Write a minimal sd-scripts-nested subset TOML (same shape build_subset emits)."""
    p = tmp / "ds.toml"
    p.write_text(
        "[general]\ncaption_extension=\".txt\"\n\n"
        "[[datasets]]\nresolution = [1024, 1024]\nbatch_size = 1\nenable_bucket = true\n"
        "  [[datasets.subsets]]\n  image_dir = \"C:/imgs/a\"\n  num_repeats = 4\n",
        encoding="utf-8",
    )
    return p


def _make_musubi(tmp: Path, scripts: list[str]) -> Path:
    d = tmp / "musubi"
    d.mkdir(exist_ok=True)
    for s in scripts:
        (d / s).write_text("# fake\n", encoding="utf-8")
    return d


def _make_sd_scripts(tmp: Path) -> Path:
    d = tmp / "sd-scripts"
    d.mkdir()
    (d / "sdxl_train_network.py").write_text("# fake\n", encoding="utf-8")
    return d


def test_sdxl_lora_no_setup_commands(tmp_path: Path):
    t = SDXLTrainer(
        sd_scripts_dir=_make_sd_scripts(tmp_path),
        venv_python=_python(tmp_path),
        pretrained_model="C:/fake", vram_gb=32.0,
    )
    cmds = t.session_setup_commands(dataset_toml=_write_subset_toml(tmp_path), run_dir=tmp_path)
    assert cmds == []


def test_zimage_lora_emits_two_cache_commands(tmp_path: Path):
    m = _make_musubi(tmp_path, ["zimage_train_network.py"])
    t = ZImageLoRATrainer(
        musubi_dir=m, venv_python=_python(tmp_path),
        dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    cmds = t.session_setup_commands(dataset_toml=_write_subset_toml(tmp_path), run_dir=tmp_path / "run")
    assert len(cmds) == 2
    assert "musubi_tuner.zimage_cache_latents" in cmds[0].cmd
    assert "musubi_tuner.zimage_cache_text_encoder_outputs" in cmds[1].cmd


def test_zimage_full_emits_two_cache_commands(tmp_path: Path):
    m = _make_musubi(tmp_path, ["zimage_train.py"])
    t = ZImageFullTrainer(
        musubi_dir=m, venv_python=_python(tmp_path),
        dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    cmds = t.session_setup_commands(dataset_toml=_write_subset_toml(tmp_path), run_dir=tmp_path / "run")
    assert len(cmds) == 2


def test_flux2_klein_emits_two_cache_commands(tmp_path: Path):
    m = _make_musubi(tmp_path, ["flux_2_train_network.py"])
    t = Flux2KleinLoRATrainer(
        musubi_dir=m, venv_python=_python(tmp_path),
        dit_path="C:/dit", vae_path="C:/vae", text_encoder_path="C:/te",
        vram_gb=32.0,
    )
    cmds = t.session_setup_commands(dataset_toml=_write_subset_toml(tmp_path), run_dir=tmp_path / "run")
    assert len(cmds) == 2
    assert "musubi_tuner.flux_2_cache_latents" in cmds[0].cmd
    assert "musubi_tuner.flux_2_cache_text_encoder_outputs" in cmds[1].cmd


def test_setup_command_passes_dataset_toml_and_paths(tmp_path: Path):
    m = _make_musubi(tmp_path, ["zimage_train_network.py"])
    t = ZImageLoRATrainer(
        musubi_dir=m, venv_python=_python(tmp_path),
        dit_path="C:/dit", vae_path="C:/vae-path", text_encoder_path="C:/te-path",
        vram_gb=32.0,
    )
    cmds = t.session_setup_commands(dataset_toml=_write_subset_toml(tmp_path), run_dir=tmp_path / "run")
    cache_lat = cmds[0]
    cache_te = cmds[1]
    assert "--vae" in cache_lat.cmd
    assert cache_lat.cmd[cache_lat.cmd.index("--vae") + 1] == "C:/vae-path"
    assert "--text_encoder" in cache_te.cmd
    assert cache_te.cmd[cache_te.cmd.index("--text_encoder") + 1] == "C:/te-path"
    assert "--dataset_config" in cache_lat.cmd
    assert "--dataset_config" in cache_te.cmd
