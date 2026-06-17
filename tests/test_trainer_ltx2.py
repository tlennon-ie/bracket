"""Tests for the LTX-2 (Lightricks native ltx-trainer) LoRA adapter.

We never launch the real trainer or touch a GPU. We verify:
  - search-space declaration (knob types + choices)
  - baseline + curated configs validate against the search space
  - config_from_dict roundtrips
  - prepare_run WRITES a config.yaml (YAML-driven, not CLI flags) and the
    returned LaunchSpec uses loss_source="stdout_log"
  - T2V vs I2V differ only in training_strategy.video.conditions
  - session_setup_commands emits the process_dataset.py pre-cache command
  - the dataset-bridge (build_ltx2_dataset_json / preprocessed_root_for)
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bracket.dataset.ltx2_dataset import (
    build_ltx2_dataset_json, preprocessed_root_for,
)
from bracket.search.space import CategoricalKnob, FloatKnob
from bracket.trainer.ltx2_lora import LTX2LoRAConfig, LTX2LoRATrainer


# ─────────────────────────── fixtures ───────────────────────────


@pytest.fixture
def fake_ltx_trainer_dir(tmp_path: Path) -> Path:
    """Pretend ltx-trainer package dir with the expected entrypoint scripts."""
    d = tmp_path / "ltx-trainer"
    (d / "scripts").mkdir(parents=True)
    (d / "scripts" / "train.py").write_text("# placeholder\n", encoding="utf-8")
    (d / "scripts" / "process_dataset.py").write_text("# placeholder\n", encoding="utf-8")
    return d


def _trainer(trainer_dir: Path, mode: str = "t2v") -> LTX2LoRATrainer:
    return LTX2LoRATrainer(
        trainer_dir=trainer_dir,
        model_path="C:/fake/ltx2-model",
        text_encoder_path="C:/fake/gemma-te",
        mode=mode,
        vram_gb=32.0,
    )


def _write_source_toml(tmp: Path, media_dir: Path) -> Path:
    p = tmp / "ds.toml"
    p.write_text(
        "[general]\ncaption_extension=\".txt\"\n\n"
        f"[[datasets]]\nresolution = [768, 768]\nbatch_size = 1\n"
        f"  [[datasets.subsets]]\n  image_dir = {str(media_dir)!r}\n  num_repeats = 1\n",
        encoding="utf-8",
    )
    return p


def _media_dir_with_clips(tmp: Path) -> Path:
    md = tmp / "clips"
    md.mkdir()
    (md / "clip1.mp4").write_bytes(b"\x00")
    (md / "clip1.txt").write_text("a cat playing", encoding="utf-8")
    (md / "clip2.mp4").write_bytes(b"\x00")
    (md / "clip2.txt").write_text("a dog running", encoding="utf-8")
    return md


# ─────────────────────────── init validation ───────────────────────────


def test_init_rejects_missing_train_script(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LTX2LoRATrainer(
            trainer_dir=tmp_path / "nope",
            model_path="C:/m", text_encoder_path="C:/te",
        )


def test_init_rejects_bad_mode(fake_ltx_trainer_dir: Path):
    with pytest.raises(ValueError):
        LTX2LoRATrainer(
            trainer_dir=fake_ltx_trainer_dir,
            model_path="C:/m", text_encoder_path="C:/te", mode="x2v",
        )


def test_name_includes_mode(fake_ltx_trainer_dir: Path):
    assert _trainer(fake_ltx_trainer_dir, "t2v").name == "ltx2-lora-t2v"
    assert _trainer(fake_ltx_trainer_dir, "i2v").name == "ltx2-lora-i2v"


# ─────────────────────────── search space ───────────────────────────


def test_search_space_has_expected_knobs(fake_ltx_trainer_dir: Path):
    space = _trainer(fake_ltx_trainer_dir).declare_search_space()
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    assert isinstance(space.knobs["lora_rank"], CategoricalKnob)
    assert tuple(space.knobs["lora_rank"].choices) == (16, 32, 64)
    assert isinstance(space.knobs["lora_alpha"], CategoricalKnob)
    assert tuple(space.knobs["lora_alpha"].choices) == (16, 32, 64)
    assert isinstance(space.knobs["scheduler_type"], CategoricalKnob)
    assert set(space.knobs["scheduler_type"].choices) == {
        "constant", "linear", "cosine", "cosine_with_restarts",
    }
    assert isinstance(space.knobs["optimizer_type"], CategoricalKnob)
    assert set(space.knobs["optimizer_type"].choices) == {"adamw", "adamw8bit"}


def test_baseline_config_validates_against_search_space(fake_ltx_trainer_dir: Path):
    t = _trainer(fake_ltx_trainer_dir)
    space = t.declare_search_space()
    baseline = t.baseline_config().to_dict()
    knobs = {k: v for k, v in baseline.items() if k in space.knobs}
    space.validate(knobs)


def test_curated_configs_validate_against_search_space(fake_ltx_trainer_dir: Path):
    t = _trainer(fake_ltx_trainer_dir)
    space = t.declare_search_space()
    curated = t.curated_configs()
    assert len(curated) >= 1
    for cfg in curated:
        knobs = {k: v for k, v in cfg.to_dict().items() if k in space.knobs}
        space.validate(knobs)


def test_config_from_dict_roundtrips(fake_ltx_trainer_dir: Path):
    t = _trainer(fake_ltx_trainer_dir)
    knobs = {
        "learning_rate": 5e-5, "optimizer_type": "adamw8bit",
        "scheduler_type": "cosine", "lora_rank": 16, "lora_alpha": 32,
        "lora_dropout": 0.0, "batch_size": 1, "gradient_accumulation_steps": 1,
        "max_grad_norm": 1.0, "mixed_precision_mode": "bf16",
        "enable_gradient_checkpointing": True, "num_dataloader_workers": 2,
    }
    cfg = t.config_from_dict(knobs)
    assert cfg.learning_rate == 5e-5
    assert cfg.optimizer_type == "adamw8bit"
    assert cfg.scheduler_type == "cosine"
    assert cfg.lora_rank == 16
    assert cfg.lora_alpha == 32


# ─────────────────────────── prepare_run / YAML ───────────────────────────


def _prepare(t: LTX2LoRATrainer, tmp_path: Path, *, max_steps: int = 200,
             with_prompts: bool = True, sample_every: int | None = 50) -> tuple:
    source_toml = _write_source_toml(tmp_path, _media_dir_with_clips(tmp_path))
    sp: Path | None = None
    if with_prompts:
        sp = tmp_path / "prompts.txt"
        sp.write_text(
            "# a comment\n"
            "a serene mountain lake --w 768 --h 768\n"
            "\n"
            "city street at night\n",
            encoding="utf-8",
        )
    run_dir = tmp_path / "run"
    spec = t.prepare_run(
        run_dir=run_dir, config=t.baseline_config(), dataset_toml=source_toml,
        max_steps=max_steps, seed=42, sample_prompts=sp,
        sample_every_n_steps=sample_every,
    )
    config_yaml = run_dir / "config.yaml"
    data = yaml.safe_load(config_yaml.read_text(encoding="utf-8"))
    return spec, data, run_dir


def test_prepare_run_writes_config_yaml(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    spec, data, run_dir = _prepare(t, tmp_path)
    assert (run_dir / "config.yaml").exists()
    assert data["model"]["training_mode"] == "lora"
    assert data["model"]["model_path"] == "C:/fake/ltx2-model"
    assert data["model"]["text_encoder_path"] == "C:/fake/gemma-te"
    assert data["optimization"]["steps"] == 200
    cfg = t.baseline_config()
    assert data["optimization"]["learning_rate"] == cfg.learning_rate
    assert data["lora"]["rank"] == cfg.lora_rank
    assert data["data"]["preprocessed_data_root"]
    # output_dir is under the run_dir
    assert str(run_dir) in str(Path(data["output_dir"]))


def test_prepare_run_validation_samples_from_prompts(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    _spec, data, _run = _prepare(t, tmp_path, sample_every=50)
    samples = data["validation"]["samples"]
    assert len(samples) == 2  # two non-comment, non-blank prompt lines
    assert samples[0]["prompt"] == "a serene mountain lake"
    assert samples[1]["prompt"] == "city street at night"
    assert data["validation"]["interval"] == 50


def test_prepare_run_interval_none_when_no_sampling(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    _spec, data, _run = _prepare(t, tmp_path, with_prompts=False, sample_every=None)
    assert data["validation"]["interval"] is None


def test_prepare_run_launchspec_shape(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    spec, _data, run_dir = _prepare(t, tmp_path)
    assert spec.loss_source == "stdout_log"
    assert spec.tfevents_glob == ""
    assert "--disable-progress-bars" in spec.cmd
    assert any(c.endswith("scripts/train.py") or c.endswith("scripts\\train.py")
               or "train.py" in c for c in spec.cmd)
    assert "train.py" in " ".join(spec.cmd)
    assert spec.cwd == fake_ltx_trainer_dir.resolve()
    assert spec.sample_dir is not None
    assert str(spec.sample_dir).endswith("samples")
    assert "PYTHONIOENCODING" in spec.env
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()


def test_t2v_has_no_first_frame_condition(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    _spec, data, _run = _prepare(t, tmp_path)
    video = data["training_strategy"]["video"]
    assert "conditions" not in video or not video.get("conditions")


def test_i2v_has_first_frame_condition(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "i2v")
    _spec, data, _run = _prepare(t, tmp_path)
    conditions = data["training_strategy"]["video"]["conditions"]
    assert any(c.get("type") == "first_frame" for c in conditions)


def test_prepare_run_resume_maps_to_load_checkpoint(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    source_toml = _write_source_toml(tmp_path, _media_dir_with_clips(tmp_path))
    resume = tmp_path / "prior.safetensors"
    resume.write_bytes(b"\x00")
    run_dir = tmp_path / "run"
    t.prepare_run(
        run_dir=run_dir, config=t.baseline_config(), dataset_toml=source_toml,
        max_steps=10, seed=0, resume_from=resume,
    )
    data = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert data["model"]["load_checkpoint"] == str(resume)


# ─────────────────────────── session setup ───────────────────────────


def test_session_setup_runs_process_dataset(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    media = _media_dir_with_clips(tmp_path)
    source_toml = _write_source_toml(tmp_path, media)
    run_dir = tmp_path / "run"
    cmds = t.session_setup_commands(dataset_toml=source_toml, run_dir=run_dir)
    assert len(cmds) == 1
    cmd = cmds[0].cmd
    assert "process_dataset.py" in " ".join(cmd)
    assert "--resolution-buckets" in cmd
    assert "--model-path" in cmd
    assert cmd[cmd.index("--model-path") + 1] == "C:/fake/ltx2-model"
    assert "--text-encoder-path" in cmd
    assert cmd[cmd.index("--text-encoder-path") + 1] == "C:/fake/gemma-te"
    assert "--output-dir" in cmd
    # The dataset.json the command references must have been written.
    json_arg = next(a for a in cmd if a.endswith(".json"))
    assert Path(json_arg).exists()
    assert cmds[0].cwd == fake_ltx_trainer_dir.resolve()


def test_session_setup_resolution_buckets_from_toml(fake_ltx_trainer_dir: Path, tmp_path: Path):
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    media = _media_dir_with_clips(tmp_path)
    source_toml = _write_source_toml(tmp_path, media)  # resolution = [768, 768]
    cmds = t.session_setup_commands(dataset_toml=source_toml, run_dir=tmp_path / "run")
    cmd = cmds[0].cmd
    buckets = cmd[cmd.index("--resolution-buckets") + 1]
    assert buckets == "768x768x49"


# ─────────────────────────── dataset bridge ───────────────────────────


def test_build_ltx2_dataset_json(tmp_path: Path):
    media = _media_dir_with_clips(tmp_path)
    source_toml = _write_source_toml(tmp_path, media)
    target = tmp_path / "dataset.json"
    out = build_ltx2_dataset_json(source_toml, target)
    assert out == target
    import json
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert len(rows) == 2
    captions = {r["caption"] for r in rows}
    assert captions == {"a cat playing", "a dog running"}
    for r in rows:
        assert r["video"].endswith(".mp4")
        assert Path(r["video"]).is_absolute()


def test_build_ltx2_dataset_json_missing_caption_is_empty(tmp_path: Path):
    media = tmp_path / "clips"
    media.mkdir()
    (media / "solo.mp4").write_bytes(b"\x00")  # no sidecar caption
    source_toml = _write_source_toml(tmp_path, media)
    target = tmp_path / "dataset.json"
    build_ltx2_dataset_json(source_toml, target)
    import json
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["caption"] == ""


def test_preprocessed_root_is_deterministic(tmp_path: Path):
    media = _media_dir_with_clips(tmp_path)
    source_toml = _write_source_toml(tmp_path, media)
    a = preprocessed_root_for(source_toml)
    b = preprocessed_root_for(source_toml)
    assert a == b
    assert a.name == ".precomputed_ltx2"
    assert a.parent == source_toml.resolve().parent


def test_session_setup_and_prepare_share_preprocessed_root(fake_ltx_trainer_dir: Path, tmp_path: Path):
    """The one preprocessing pass must be reused by every candidate: the
    process_dataset --output-dir and the training config's
    preprocessed_data_root must point at the same deterministic dir."""
    t = _trainer(fake_ltx_trainer_dir, "t2v")
    media = _media_dir_with_clips(tmp_path)
    source_toml = _write_source_toml(tmp_path, media)
    cmds = t.session_setup_commands(dataset_toml=source_toml, run_dir=tmp_path / "setup")
    setup_out = cmds[0].cmd[cmds[0].cmd.index("--output-dir") + 1]
    run_dir = tmp_path / "run"
    t.prepare_run(
        run_dir=run_dir, config=t.baseline_config(), dataset_toml=source_toml,
        max_steps=10, seed=0,
    )
    data = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert Path(data["data"]["preprocessed_data_root"]) == Path(setup_out)
