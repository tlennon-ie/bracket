"""Tests for dataset_has_cached_latents — the heuristic that lets the
sd-scripts trainers add ``--skip_cache_check`` when the dataset is
already cached on disk."""
from __future__ import annotations

import tempfile
from pathlib import Path

import toml

from bracket.dataset.latent_cache import dataset_has_cached_latents


def _make_image_dir(
    base: Path,
    name: str,
    *,
    n_images: int,
    n_cached: int,
) -> Path:
    """Create a directory with ``n_images`` placeholder images, of which
    the first ``n_cached`` have a matching sd-scripts-style ``.npz``
    neighbour (``<stem>_1024x1024_sdxl.npz``)."""
    d = base / name
    d.mkdir(parents=True)
    for i in range(n_images):
        stem = f"img_{i:04d}"
        (d / f"{stem}.jpg").write_bytes(b"")
        (d / f"{stem}.txt").write_text(f"caption for {stem}")
        if i < n_cached:
            (d / f"{stem}_1024x1024_sdxl.npz").write_bytes(b"")
    return d


def _write_dataset_toml(base: Path, image_dirs: list[Path]) -> Path:
    data = {
        "datasets": [
            {
                "batch_size": 2,
                "resolution": [1024, 1024],
                "subsets": [
                    {"image_dir": str(d), "num_repeats": 1, "caption_extension": ".txt"}
                    for d in image_dirs
                ],
            }
        ]
    }
    p = base / "dataset.toml"
    p.write_text(toml.dumps(data))
    return p


def test_detects_fully_cached_dataset():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d = _make_image_dir(base, "all_cached", n_images=20, n_cached=20)
        toml_path = _write_dataset_toml(base, [d])
        assert dataset_has_cached_latents(toml_path) is True


def test_skips_dataset_with_no_cache():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d = _make_image_dir(base, "cold", n_images=20, n_cached=0)
        toml_path = _write_dataset_toml(base, [d])
        assert dataset_has_cached_latents(toml_path) is False


def test_partial_cache_below_threshold_is_cold():
    """Half-cached datasets shouldn't add ``--skip_cache_check`` —
    that flag is only worth it when the rebuild would be a no-op."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d = _make_image_dir(base, "half", n_images=20, n_cached=10)
        toml_path = _write_dataset_toml(base, [d])
        assert dataset_has_cached_latents(toml_path) is False


def test_aggregates_across_subsets():
    """A dataset with multiple image_dirs is hot only when the union is
    above threshold — one cold dir among warm ones is fine."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d1 = _make_image_dir(base, "warm_a", n_images=50, n_cached=50)
        d2 = _make_image_dir(base, "warm_b", n_images=50, n_cached=48)
        toml_path = _write_dataset_toml(base, [d1, d2])
        assert dataset_has_cached_latents(toml_path) is True


def test_missing_image_dir_does_not_crash():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        toml_path = _write_dataset_toml(base, [base / "does_not_exist"])
        assert dataset_has_cached_latents(toml_path) is False


def test_dataset_with_no_images_is_cold():
    """An empty dataset shouldn't be reported as cached."""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d = base / "empty"
        d.mkdir()
        toml_path = _write_dataset_toml(base, [d])
        assert dataset_has_cached_latents(toml_path) is False


def test_threshold_override():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        d = _make_image_dir(base, "two_thirds", n_images=30, n_cached=20)
        toml_path = _write_dataset_toml(base, [d])
        assert dataset_has_cached_latents(toml_path, threshold=0.95) is False
        assert dataset_has_cached_latents(toml_path, threshold=0.5) is True
