"""Tests for ai-toolkit video (LTX-2.5, MiniMax-H3) and audio (ACE-Step) LoRA.

The image path is covered by ``test_trainer_aitk.py``; this file covers what
the ``media_kind`` split adds. We never launch the real trainer or touch a GPU.

The load-bearing assertions here are the ones that would silently produce a
*wrong but running* job rather than a crash:

  - ``datasets[0].num_frames > 1`` for every video profile. ai-toolkit's own
    default is 1, which makes its dataloader read stills — a video LoRA would
    train happily and learn no motion at all.
  - the guidance-distilled MiniMax checkpoints carry BOTH the contrastive
    guidance loss and the training adapter; training without them degrades the
    distillation with no error.
  - audio profiles emit no pixel ``resolution`` bucket list.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bracket.trainer.aitk_lora import AiToolkitLoRATrainer
from bracket.trainer.aitk_profiles import (
    MEDIA_AUDIO,
    MEDIA_IMAGE,
    MEDIA_VIDEO,
    PROFILES,
    AiToolkitProfile,
    get_profile,
)

_VIDEO_IDS = ("ltx25", "minimax_h3", "minimax_h3_ref2va")
_AUDIO_IDS = ("ace_step_15", "ace_step_15_xl")
_MINIMAX_IDS = ("minimax_h3", "minimax_h3_ref2va")


# ─────────────────────────── fixtures ───────────────────────────


@pytest.fixture
def fake_aitk_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ai-toolkit"
    d.mkdir(parents=True)
    (d / "run.py").write_text("# placeholder\n", encoding="utf-8")
    return d


def _trainer(aitk_dir: Path, profile_id: str) -> AiToolkitLoRATrainer:
    """Build the adapter the way ``registry._build_aitk_lora`` does."""
    p = get_profile(profile_id)
    return AiToolkitLoRATrainer(
        aitk_dir=aitk_dir,
        venv_python="C:/fake/aitk-venv/python.exe",
        model_name_or_path=p.default_model,
        model_id=p.model_id,
        model_extra=p.model_extra,
        media_kind=p.media_kind,
        train_extra=p.train_extra,
        dataset_extra=p.dataset_extra,
        network_extra=p.network_extra,
        sample_extra=p.sample_extra,
        vram_gb=32.0,
    )


def _source_toml(tmp: Path) -> Path:
    media = tmp / "clips"
    media.mkdir(parents=True, exist_ok=True)
    (media / "a.mp4").write_bytes(b"\x00")
    (media / "a.txt").write_text("a cat", encoding="utf-8")
    p = tmp / "ds.toml"
    p.write_text(
        '[general]\ncaption_extension=".txt"\n\n'
        "[[datasets]]\nresolution = [768, 768]\nbatch_size = 1\n"
        f"  [[datasets.subsets]]\n  image_dir = {str(media)!r}\n  num_repeats = 1\n",
        encoding="utf-8",
    )
    return p


def _process(aitk_dir: Path, tmp_path: Path, profile_id: str) -> dict:
    t = _trainer(aitk_dir, profile_id)
    run_dir = tmp_path / f"run-{profile_id}"
    t.prepare_run(
        run_dir=run_dir, config=t.baseline_config(),
        dataset_toml=_source_toml(tmp_path / f"ds-{profile_id}"),
        max_steps=200, seed=42,
    )
    doc = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    return doc["config"]["process"][0]


# ─────────────────────────── profile registry ───────────────────────────


def test_new_profiles_are_registered():
    for pid in (*_VIDEO_IDS, *_AUDIO_IDS):
        assert pid in PROFILES, f"missing ai-toolkit profile {pid}"


def test_every_profile_declares_an_arch_selector():
    """Each profile must select an architecture — via ``arch`` or a boolean.

    A profile with neither would silently fall back to ai-toolkit's default
    architecture and train the wrong model.
    """
    for pid, p in PROFILES.items():
        selectors = [k for k in p.model_extra if k == "arch" or k.startswith("is_")]
        assert selectors, f"{pid} declares no architecture selector"


def test_profile_arch_names_match_ai_toolkit():
    """Arch strings are ai-toolkit's, not Bracket's ``model_id``."""
    assert get_profile("ltx25").model_extra["arch"] == "ltx2.5"
    assert get_profile("minimax_h3").model_extra["arch"] == "minimax_h3"
    assert get_profile("minimax_h3_ref2va").model_extra["arch"] == "minimax_h3_ref2va"
    assert get_profile("ace_step_15").model_extra["arch"] == "ace_step_15"
    assert get_profile("ace_step_15_xl").model_extra["arch"] == "ace_step_15_xl"


def test_media_kinds_are_as_declared():
    for pid in _VIDEO_IDS:
        assert get_profile(pid).media_kind == MEDIA_VIDEO
    for pid in _AUDIO_IDS:
        assert get_profile(pid).media_kind == MEDIA_AUDIO
    assert get_profile("chroma").media_kind == MEDIA_IMAGE


def test_unknown_profile_raises_with_known_ids_listed():
    with pytest.raises(KeyError) as exc:
        get_profile("not-a-model")
    assert "ltx25" in str(exc.value)


def test_profile_rejects_bad_media_kind():
    with pytest.raises(ValueError):
        AiToolkitProfile(model_id="x", default_model="y", media_kind="hologram")


def test_adapter_rejects_bad_media_kind(fake_aitk_dir: Path):
    with pytest.raises(ValueError):
        AiToolkitLoRATrainer(
            aitk_dir=fake_aitk_dir, venv_python="py",
            model_name_or_path="m", model_id="x", model_extra={"arch": "x"},
            media_kind="hologram",
        )


# ─────────────────────────── video ───────────────────────────


@pytest.mark.parametrize("profile_id", _VIDEO_IDS)
def test_video_dataset_trains_on_clips_not_stills(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    """``num_frames`` must be > 1 or ai-toolkit reads images, not video."""
    ds = _process(fake_aitk_dir, tmp_path, profile_id)["datasets"][0]
    assert ds["num_frames"] > 1, (
        f"{profile_id}: num_frames <= 1 makes ai-toolkit's dataloader read "
        "stills — the video LoRA would learn no motion"
    )
    assert ds["fps"] == 24
    assert ds["do_audio"] is True
    assert ds["cache_latents_to_disk"] is True


@pytest.mark.parametrize("profile_id", _VIDEO_IDS)
def test_video_samples_render_clips(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    sample = _process(fake_aitk_dir, tmp_path, profile_id)["sample"]
    assert sample["num_frames"] > 1
    assert sample["fps"] == 24
    # Video samples render at 768², not the image default of 1024².
    assert sample["width"] == 768
    assert sample["height"] == 768


def test_ltx25_frame_count_sits_on_the_vae_grid(
    fake_aitk_dir: Path, tmp_path: Path,
):
    """LTX-2's VAE quantises clip length onto an 8n+1 grid."""
    ds = _process(fake_aitk_dir, tmp_path, "ltx25")["datasets"][0]
    assert (ds["num_frames"] - 1) % 8 == 0, ds["num_frames"]
    sample = _process(fake_aitk_dir, tmp_path, "ltx25")["sample"]
    assert (sample["num_frames"] - 1) % 8 == 0, sample["num_frames"]


@pytest.mark.parametrize("profile_id", _MINIMAX_IDS)
def test_minimax_frame_count_sits_on_the_vae_grid(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    """MiniMax-H3's VAE quantises clip length onto a 17n+5 grid."""
    ds = _process(fake_aitk_dir, tmp_path, profile_id)["datasets"][0]
    assert (ds["num_frames"] - 5) % 17 == 0, ds["num_frames"]


@pytest.mark.parametrize("profile_id", _MINIMAX_IDS)
def test_minimax_handles_guidance_distillation(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    """Both remedies must be on: contrastive guidance AND the training adapter.

    MiniMax-H3 is guidance-distilled. Training it without these produces a
    LoRA that runs but degrades the base model's distillation — a silent
    quality regression, not an error.
    """
    proc = _process(fake_aitk_dir, tmp_path, profile_id)
    assert proc["train"]["do_guidance_loss"] is True
    assert proc["train"]["guidance_loss_target"] == 3.5
    adapter = proc["model"]["assistant_lora_path"]
    assert adapter.startswith("ostris/minimax_h3_training_adapter/")
    # ref2va needs its own adapter — the plain-H3 one is the wrong partition.
    if profile_id == "minimax_h3_ref2va":
        assert "ref2va" in adapter
    else:
        assert "ref2va" not in adapter


@pytest.mark.parametrize("profile_id", _MINIMAX_IDS)
def test_minimax_excludes_adaln_from_lora(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    net = _process(fake_aitk_dir, tmp_path, profile_id)["network"]
    assert net["network_kwargs"]["ignore_if_contains"] == ["adaln_proj"]
    # Searchable rank/alpha still flow through alongside the arch extras.
    assert net["linear"] == 16
    assert net["linear_alpha"] == 16


@pytest.mark.parametrize("profile_id", _MINIMAX_IDS)
def test_minimax_samples_at_distilled_guidance(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    """A distilled model samples at guidance 1, not the image default of 4."""
    sample = _process(fake_aitk_dir, tmp_path, profile_id)["sample"]
    assert sample["guidance_scale"] == 1


def test_video_profiles_use_prequantized_qtypes(
    fake_aitk_dir: Path, tmp_path: Path,
):
    """qtypes must match the shipped checkpoints or every run re-quantises."""
    h3 = _process(fake_aitk_dir, tmp_path, "minimax_h3")["model"]
    assert h3["qtype"] == "convrot8"
    assert h3["qtype_te"] == "nvfp4"
    ltx = _process(fake_aitk_dir, tmp_path, "ltx25")["model"]
    assert ltx["qtype"] == "convrot8"
    assert ltx["qtype_te"] == "convrot8"


def test_auto_frame_count_requires_batch_size_one(fake_aitk_dir: Path):
    """ai-toolkit raises if ``auto_frame_count`` meets ``batch_size > 1``."""
    for pid in _VIDEO_IDS:
        p = get_profile(pid)
        if p.dataset_extra.get("auto_frame_count"):
            t = _trainer(fake_aitk_dir, pid)
            assert t.baseline_config().batch_size == 1
            assert t.declare_search_space().knobs["batch_size"].value == 1


# ─────────────────────────── audio ───────────────────────────


@pytest.mark.parametrize("profile_id", _AUDIO_IDS)
def test_audio_dataset_has_no_pixel_resolution(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    ds = _process(fake_aitk_dir, tmp_path, profile_id)["datasets"][0]
    assert "resolution" not in ds
    assert "num_frames" not in ds
    assert ds["cache_latents_to_disk"] is True


@pytest.mark.parametrize("profile_id", _AUDIO_IDS)
def test_audio_keeps_text_encoder_resident(
    fake_aitk_dir: Path, tmp_path: Path, profile_id: str,
):
    """ACE-Step conditions on the caption each step — the TE cannot unload."""
    train = _process(fake_aitk_dir, tmp_path, profile_id)["train"]
    assert train["unload_text_encoder"] is False
    assert train["timestep_type"] == "linear"


def test_ace_step_xl_points_at_the_xl_checkpoint():
    base = get_profile("ace_step_15").default_model
    xl = get_profile("ace_step_15_xl").default_model
    assert "xl" in xl and "xl" not in base
    assert base != xl


# ─────────────────────── image path is unchanged ───────────────────────


def test_image_profile_still_emits_resolution_and_no_video_keys(
    fake_aitk_dir: Path, tmp_path: Path,
):
    """Regression guard: the media split must not alter the image path."""
    ds = _process(fake_aitk_dir, tmp_path, "chroma")["datasets"][0]
    assert ds["resolution"] == [512, 768, 1024]
    for key in ("num_frames", "fps", "do_audio", "auto_frame_count"):
        assert key not in ds
    sample = _process(fake_aitk_dir, tmp_path, "chroma")["sample"]
    assert sample["width"] == 1024
    assert sample["height"] == 1024
    assert sample["guidance_scale"] == 4


def test_profile_dicts_are_not_mutated_by_config_assembly(
    fake_aitk_dir: Path, tmp_path: Path,
):
    """The adapter copies defensively — profiles are module-level singletons."""
    before = dict(get_profile("minimax_h3").train_extra)
    _process(fake_aitk_dir, tmp_path, "minimax_h3")
    assert dict(get_profile("minimax_h3").train_extra) == before
