"""Tests for the dataset subset generator. Uses pillow to fabricate tiny PNGs
since sd-scripts decides bucket eligibility from real image dimensions, but
since we don't actually run the trainer here we only need files with the
right extensions and caption sidecars."""
from __future__ import annotations

from pathlib import Path

import pytest
import toml

from bracket.dataset.subset import DatasetSubsetSpec, build_subset


def _make_dummy_dataset(root: Path, name: str, n_images: int) -> Path:
    img_dir = root / name
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        (img_dir / f"img_{i:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
        (img_dir / f"img_{i:03d}.txt").write_text(f"caption {i}", encoding="utf-8")
    return img_dir


def test_build_subset_writes_derived_toml(tmp_path: Path):
    src_root = tmp_path / "src"
    a = _make_dummy_dataset(src_root, "professional", 50)
    b = _make_dummy_dataset(src_root, "amateur", 30)
    src_toml = tmp_path / "dataset.toml"
    src_toml.write_text(toml.dumps({
        "general": {"resolution": [1024, 1024], "caption_extension": ".txt", "batch_size": 1, "enable_bucket": True},
        "datasets": [
            {"image_directory": str(a), "cache_directory": str(src_root / "cache_a"), "num_repeats": 4},
            {"image_directory": str(b), "cache_directory": str(src_root / "cache_b"), "num_repeats": 16},
        ],
    }), encoding="utf-8")
    out_toml = build_subset(
        source_toml=src_toml,
        target_dir=tmp_path / "subset",
        spec=DatasetSubsetSpec(images_per_dataset=10, seed=0),
    )
    derived = toml.load(out_toml)
    assert derived["general"]["caption_extension"] == ".txt"
    # sd-scripts nested format: one [[datasets]] block with N [[subsets]]
    assert len(derived["datasets"]) == 1
    ds = derived["datasets"][0]
    assert ds["resolution"] == [1024, 1024]
    assert ds["enable_bucket"] is True
    assert "subsets" in ds
    subsets = ds["subsets"]
    assert len(subsets) == 2
    repeats = sorted(int(s["num_repeats"]) for s in subsets)
    assert repeats == [4, 16]
    for s in subsets:
        sd = Path(s["image_dir"])
        assert len(list(sd.glob("*.png"))) == 10
        assert len(list(sd.glob("*.txt"))) == 10


def test_build_subset_handles_dataset_smaller_than_k(tmp_path: Path):
    src = tmp_path / "src"
    _make_dummy_dataset(src, "tiny", 3)
    src_toml = tmp_path / "dataset.toml"
    src_toml.write_text(toml.dumps({
        "general": {"caption_extension": ".txt"},
        "datasets": [{"image_directory": str(src / "tiny")}],
    }), encoding="utf-8")
    out_toml = build_subset(
        source_toml=src_toml, target_dir=tmp_path / "subset",
        spec=DatasetSubsetSpec(images_per_dataset=10, seed=0),
    )
    derived = toml.load(out_toml)
    pngs = list(Path(derived["datasets"][0]["subsets"][0]["image_dir"]).glob("*.png"))
    assert len(pngs) == 3  # took everything available


def test_build_subset_seed_is_deterministic(tmp_path: Path):
    _make_dummy_dataset(tmp_path / "src", "ds", 50)
    src_toml = tmp_path / "dataset.toml"
    src_toml.write_text(toml.dumps({
        "general": {"caption_extension": ".txt"},
        "datasets": [{"image_directory": str(tmp_path / "src" / "ds")}],
    }), encoding="utf-8")
    out1 = build_subset(src_toml, tmp_path / "s1", spec=DatasetSubsetSpec(images_per_dataset=5, seed=7))
    out2 = build_subset(src_toml, tmp_path / "s2", spec=DatasetSubsetSpec(images_per_dataset=5, seed=7))
    a = sorted(p.name for p in Path(toml.load(out1)["datasets"][0]["subsets"][0]["image_dir"]).glob("*.png"))
    b = sorted(p.name for p in Path(toml.load(out2)["datasets"][0]["subsets"][0]["image_dir"]).glob("*.png"))
    assert a == b


def test_build_subset_full_dataset_normalises_without_copying(tmp_path: Path):
    """images_per_dataset=0 + copy_files=False is the 'use full dataset'
    flow: convert musubi-tuner's flat ``image_directory`` schema into
    sd-scripts' nested ``[[datasets.subsets]].image_dir`` schema without
    duplicating image files to a subset dir. This is the fix for sd-scripts
    rejecting musubi-format TOMLs with 'extra keys not allowed @
    data["datasets"][0]["image_directory"]'."""

    src_root = tmp_path / "src"
    a = _make_dummy_dataset(src_root, "class_a", 25)
    src_toml = tmp_path / "dataset.toml"
    src_toml.write_text(toml.dumps({
        "general": {"resolution": [1024, 1024], "caption_extension": ".txt"},
        "datasets": [{"image_directory": str(a), "num_repeats": 2}],
    }), encoding="utf-8")
    out_toml = build_subset(
        source_toml=src_toml,
        target_dir=tmp_path / "subset",
        spec=DatasetSubsetSpec(
            images_per_dataset=0, seed=0, copy_files=False,
        ),
    )
    derived = toml.load(out_toml)
    # Output uses sd-scripts schema (no top-level image_directory).
    ds = derived["datasets"][0]
    assert "image_directory" not in ds
    subsets = ds["subsets"]
    assert len(subsets) == 1
    # No file copy: the toml points at the ORIGINAL directory.
    image_dir = Path(subsets[0]["image_dir"])
    assert image_dir.resolve() == a.resolve()
    # No partial subset was written elsewhere.
    assert not (tmp_path / "subset" / "d00_00_class_a").exists()


def test_build_subset_missing_image_dir_raises(tmp_path: Path):
    src_toml = tmp_path / "dataset.toml"
    src_toml.write_text(toml.dumps({
        "general": {},
        "datasets": [{"image_directory": str(tmp_path / "missing")}],
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        build_subset(src_toml, tmp_path / "out")
