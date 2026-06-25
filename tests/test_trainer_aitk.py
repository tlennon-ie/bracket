"""Tests for the ai-toolkit (ostris) LoRA adapter + dataset bridge.

We never launch the real trainer or touch a GPU. We verify:
  - search-space declaration (knob types + choices)
  - baseline + curated configs validate against the search space
  - config_from_dict roundtrips
  - prepare_run WRITES a config.yaml (YAML-driven, not CLI flags) with the
    ai-toolkit job/process shape, the right arch per preset, and
    logging.use_ui_logger == true
  - the returned LaunchSpec uses loss_source="sqlite_db" + loss_db_path,
    cmd contains run.py + the config path, cwd == aitk_dir, tfevents_glob == ""
  - arch parametrization across ≥2 archs (chroma + omnigen2)
  - the dataset bridge (image_dir_for)
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bracket.dataset.aitk_dataset import image_dir_for
from bracket.search.space import CategoricalKnob, FloatKnob
from bracket.trainer.aitk_lora import AiToolkitLoRAConfig, AiToolkitLoRATrainer


# ─────────────────────────── fixtures ───────────────────────────


@pytest.fixture
def fake_aitk_dir(tmp_path: Path) -> Path:
    """Pretend ai-toolkit checkout dir with the expected run.py entrypoint."""
    d = tmp_path / "ai-toolkit"
    d.mkdir(parents=True)
    (d / "run.py").write_text("# placeholder\n", encoding="utf-8")
    return d


def _trainer(
    aitk_dir: Path,
    *,
    model_id: str = "chroma",
    model: str = "lodestones/Chroma",
) -> AiToolkitLoRATrainer:
    return AiToolkitLoRATrainer(
        aitk_dir=aitk_dir,
        venv_python="C:/fake/aitk-venv/python.exe",
        model_name_or_path=model,
        model_id=model_id,
        model_extra=_MODEL_EXTRA[model_id],
        vram_gb=24.0,
    )


# Per-model architecture selector + quantize mode, verbatim from each model's
# ai-toolkit example config. chroma/omnigen2/flex2 use ``arch``; lumina2 uses
# ``is_lumina2``; flex1 uses ``is_flux``; quantize vs quantize_te differs.
_MODEL_EXTRA: dict[str, dict] = {
    "chroma": {"arch": "chroma", "quantize": True},
    "lumina2": {"is_lumina2": True, "quantize_te": True},
    "omnigen2": {"arch": "omnigen2", "quantize_te": True},
    "flex1": {"is_flux": True, "quantize": True},
    "flex2": {"arch": "flex2", "quantize": True, "quantize_te": True},
}


def _media_dir_with_images(tmp: Path) -> Path:
    md = tmp / "images"
    md.mkdir()
    (md / "img1.png").write_bytes(b"\x00")
    (md / "img1.txt").write_text("a cat", encoding="utf-8")
    (md / "img2.png").write_bytes(b"\x00")
    (md / "img2.txt").write_text("a dog", encoding="utf-8")
    return md


def _write_source_toml(tmp: Path, media_dir: Path) -> Path:
    p = tmp / "ds.toml"
    p.write_text(
        "[general]\ncaption_extension=\".txt\"\n\n"
        f"[[datasets]]\nresolution = [1024, 1024]\nbatch_size = 1\n"
        f"  [[datasets.subsets]]\n  image_dir = {str(media_dir)!r}\n  num_repeats = 1\n",
        encoding="utf-8",
    )
    return p


# ─────────────────────────── init validation ───────────────────────────


def test_init_rejects_missing_run_script(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        AiToolkitLoRATrainer(
            aitk_dir=tmp_path / "nope",
            venv_python="C:/fake/python.exe",
            model_name_or_path="lodestones/Chroma",
            model_id="chroma",
            model_extra={"arch": "chroma", "quantize": True},
        )


def test_name_includes_model_id(fake_aitk_dir: Path):
    assert _trainer(fake_aitk_dir, model_id="chroma").name == "aitk-lora-chroma"
    assert _trainer(fake_aitk_dir, model_id="omnigen2").name == "aitk-lora-omnigen2"


def test_lumina2_uses_is_lumina2_selector_not_arch(fake_aitk_dir: Path, tmp_path: Path):
    """Lumina2 must use the ``is_lumina2: true`` boolean (its example config's
    selector), NOT ``arch: lumina2`` — there is no lumina2 entry in
    ai-toolkit's arch registry, so ``arch: lumina2`` would be rejected."""
    md = _media_dir_with_images(tmp_path)
    toml = _write_source_toml(tmp_path, md)
    t = _trainer(fake_aitk_dir, model_id="lumina2", model="Alpha-VLLM/Lumina-Image-2.0")
    spec = t.prepare_run(
        run_dir=tmp_path / "run", config=t.baseline_config(),
        dataset_toml=toml, max_steps=100, seed=0,
    )
    import yaml as _yaml
    doc = _yaml.safe_load((tmp_path / "run" / "config.yaml").read_text(encoding="utf-8"))
    model_block = doc["config"]["process"][0]["model"]
    assert model_block["is_lumina2"] is True
    assert "arch" not in model_block
    assert model_block["quantize_te"] is True


# ─────────────────────────── search space ───────────────────────────


def test_search_space_has_expected_knobs(fake_aitk_dir: Path):
    space = _trainer(fake_aitk_dir).declare_search_space()
    assert isinstance(space.knobs["learning_rate"], FloatKnob)
    assert space.knobs["learning_rate"].log is True
    assert isinstance(space.knobs["network_rank"], CategoricalKnob)
    assert tuple(space.knobs["network_rank"].choices) == (8, 16, 32, 64)
    assert isinstance(space.knobs["network_alpha"], CategoricalKnob)
    assert tuple(space.knobs["network_alpha"].choices) == (8, 16, 32)
    assert isinstance(space.knobs["optimizer"], CategoricalKnob)
    assert set(space.knobs["optimizer"].choices) == {"adamw8bit", "adamw"}


def test_baseline_config_validates_against_search_space(fake_aitk_dir: Path):
    t = _trainer(fake_aitk_dir)
    space = t.declare_search_space()
    baseline = t.baseline_config().to_dict()
    knobs = {k: v for k, v in baseline.items() if k in space.knobs}
    space.validate(knobs)


def test_curated_configs_validate_against_search_space(fake_aitk_dir: Path):
    t = _trainer(fake_aitk_dir)
    space = t.declare_search_space()
    curated = t.curated_configs()
    assert len(curated) >= 1
    for cfg in curated:
        knobs = {k: v for k, v in cfg.to_dict().items() if k in space.knobs}
        space.validate(knobs)


def test_config_from_dict_roundtrips(fake_aitk_dir: Path):
    t = _trainer(fake_aitk_dir)
    knobs = {
        "learning_rate": 5e-5, "optimizer": "adamw",
        "network_rank": 32, "network_alpha": 16,
        "batch_size": 1, "gradient_accumulation_steps": 1,
        "dtype": "bf16", "gradient_checkpointing": True,
        "noise_scheduler": "flowmatch",
    }
    cfg = t.config_from_dict(knobs)
    assert cfg.learning_rate == 5e-5
    assert cfg.optimizer == "adamw"
    assert cfg.network_rank == 32
    assert cfg.network_alpha == 16


# ─────────────────────────── prepare_run / YAML ───────────────────────────


def _prepare(
    t: AiToolkitLoRATrainer, tmp_path: Path, *, max_steps: int = 200,
    with_prompts: bool = True, sample_every: int | None = 50,
) -> tuple:
    source_toml = _write_source_toml(tmp_path, _media_dir_with_images(tmp_path))
    sp: Path | None = None
    if with_prompts:
        sp = tmp_path / "prompts.txt"
        sp.write_text(
            "# a comment\n"
            "a serene mountain lake --w 1024 --h 1024\n"
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


def _process(data: dict) -> dict:
    return data["config"]["process"][0]


def test_prepare_run_writes_config_yaml(fake_aitk_dir: Path, tmp_path: Path):
    t = _trainer(fake_aitk_dir, model_id="chroma", model="lodestones/Chroma")
    spec, data, run_dir = _prepare(t, tmp_path)
    assert (run_dir / "config.yaml").exists()
    assert data["job"] == "extension"
    proc = _process(data)
    assert proc["type"] == "sd_trainer"
    assert proc["model"]["arch"] == "chroma"
    assert proc["model"]["name_or_path"] == "lodestones/Chroma"
    assert proc["network"]["linear"] == t.baseline_config().network_rank
    assert proc["network"]["linear_alpha"] == t.baseline_config().network_alpha
    assert proc["train"]["steps"] == 200
    assert proc["train"]["lr"] == t.baseline_config().learning_rate
    # logging.use_ui_logger MUST be true to enable the SQLite loss DB.
    assert proc["logging"]["use_ui_logger"] is True


def test_prepare_run_folder_path_from_bridge(fake_aitk_dir: Path, tmp_path: Path):
    t = _trainer(fake_aitk_dir)
    media = _media_dir_with_images(tmp_path)
    source_toml = _write_source_toml(tmp_path, media)
    run_dir = tmp_path / "run"
    t.prepare_run(
        run_dir=run_dir, config=t.baseline_config(), dataset_toml=source_toml,
        max_steps=10, seed=0,
    )
    data = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    ds = _process(data)["datasets"][0]
    assert Path(ds["folder_path"]) == media.resolve()
    assert ds["caption_ext"] == "txt"
    assert ds["cache_latents_to_disk"] is True


def test_prepare_run_sample_prompts_from_file(fake_aitk_dir: Path, tmp_path: Path):
    t = _trainer(fake_aitk_dir)
    _spec, data, _run = _prepare(t, tmp_path, sample_every=50)
    prompts = _process(data)["sample"]["prompts"]
    assert prompts == ["a serene mountain lake", "city street at night"]
    assert _process(data)["sample"]["sample_every"] == 50


def test_prepare_run_launchspec_shape(fake_aitk_dir: Path, tmp_path: Path):
    t = _trainer(fake_aitk_dir)
    spec, _data, run_dir = _prepare(t, tmp_path)
    assert spec.loss_source == "sqlite_db"
    assert spec.loss_db_path is not None
    assert str(spec.loss_db_path).endswith("loss_log.db")
    assert spec.tfevents_glob == ""
    assert spec.sample_dir is not None
    assert str(spec.sample_dir).endswith("samples")
    assert "run.py" in " ".join(spec.cmd)
    assert str(run_dir / "config.yaml") in spec.cmd
    assert spec.cwd == fake_aitk_dir.resolve()
    assert "PYTHONIOENCODING" in spec.env
    assert spec.output_dir.exists()
    assert spec.logging_dir.exists()


def test_loss_db_path_under_save_root(fake_aitk_dir: Path, tmp_path: Path):
    """loss_db_path = <run_dir>/output/candidate/loss_log.db; samples sibling."""
    t = _trainer(fake_aitk_dir)
    spec, _data, run_dir = _prepare(t, tmp_path)
    save_root = run_dir / "output" / "candidate"
    assert spec.loss_db_path == save_root / "loss_log.db"
    assert spec.sample_dir == save_root / "samples"


def test_omnigen2_arch_parametrization(fake_aitk_dir: Path, tmp_path: Path):
    """Second arch proves arch parametrisation flows into the YAML."""
    t = _trainer(fake_aitk_dir, model_id="omnigen2", model="OmniGen2/OmniGen2")
    _spec, data, _run = _prepare(t, tmp_path)
    proc = _process(data)
    assert proc["model"]["arch"] == "omnigen2"
    assert proc["model"]["name_or_path"] == "OmniGen2/OmniGen2"
    # OmniGen2 does NOT bypass the guidance embedder.
    assert "bypass_guidance_embedding" not in proc["train"]


def test_flex_arch_sets_bypass_guidance(fake_aitk_dir: Path, tmp_path: Path):
    """Flex archs require bypass_guidance_embedding during training."""
    t = _trainer(fake_aitk_dir, model_id="flex2", model="ostris/Flex.2-preview")
    _spec, data, _run = _prepare(t, tmp_path)
    proc = _process(data)
    assert proc["model"]["arch"] == "flex2"
    assert proc["train"]["bypass_guidance_embedding"] is True


def test_prepare_run_rejects_wrong_config_type(fake_aitk_dir: Path, tmp_path: Path):
    t = _trainer(fake_aitk_dir)
    source_toml = _write_source_toml(tmp_path, _media_dir_with_images(tmp_path))
    with pytest.raises(TypeError):
        t.prepare_run(
            run_dir=tmp_path / "run", config=object(),  # type: ignore[arg-type]
            dataset_toml=source_toml, max_steps=10, seed=0,
        )


# ─────────────────────────── dataset bridge ───────────────────────────


def test_image_dir_for_extracts_folder_and_caption_ext(tmp_path: Path):
    media = _media_dir_with_images(tmp_path)
    source_toml = _write_source_toml(tmp_path, media)
    folder_path, caption_ext = image_dir_for(source_toml)
    assert Path(folder_path) == media.resolve()
    assert Path(folder_path).is_absolute()
    # ai-toolkit wants the dot-less form.
    assert caption_ext == "txt"


def test_image_dir_for_flat_image_directory(tmp_path: Path):
    media = _media_dir_with_images(tmp_path)
    p = tmp_path / "flat.toml"
    p.write_text(
        "[general]\ncaption_extension=\".caption\"\n\n"
        f"[[datasets]]\nimage_directory = {str(media)!r}\nnum_repeats = 1\n",
        encoding="utf-8",
    )
    folder_path, caption_ext = image_dir_for(p)
    assert Path(folder_path) == media.resolve()
    assert caption_ext == "caption"


def test_image_dir_for_missing_toml_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        image_dir_for(tmp_path / "does-not-exist.toml")


def test_image_dir_for_no_media_raises(tmp_path: Path):
    p = tmp_path / "empty.toml"
    p.write_text("[general]\ncaption_extension=\".txt\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        image_dir_for(p)
