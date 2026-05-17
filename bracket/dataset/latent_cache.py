"""Detect existing sd-scripts latent caches alongside a dataset's images.

sd-scripts saves per-image latent caches as ``<image_stem>_<W>x<H>_<arch>.npz``
next to the source image when ``cache_latents_to_disk=True``. On a re-run
the trainer still walks the dataset image-by-image and validates each
existing cache by reading it — which on a 26k-image dataset can take many
minutes on disk-bound machines, even though every latent is already on
disk.

If we can see the caches are already there, we pass ``--skip_cache_check``
so sd-scripts uses the existing ``.npz`` without re-validating shape and
key contents. New images without a cache are still processed normally —
the flag only suppresses the integrity check, never the cache-write path.

Public API is one function: :func:`dataset_has_cached_latents`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import toml

logger = logging.getLogger(__name__)

# Image extensions sd-scripts will pick up. Match-case is irrelevant on
# Windows, but kept lower-case to be safe on Linux.
_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
)

# Minimum fraction of images that must have a matching ``.npz`` neighbour
# before we declare the dataset "already cached". Sub-threshold = treat as
# cold cache and let sd-scripts walk normally.
_CACHED_FRACTION_THRESHOLD = 0.95


def _iter_image_dirs(toml_path: Path) -> Iterable[Path]:
    """Yield every ``image_dir`` / ``image_directory`` referenced by the
    dataset TOML. Tolerates both sd-scripts-nested and musubi-flat schemas."""
    data = toml.load(Path(toml_path))
    for ds in data.get("datasets", []):
        subsets = ds.get("subsets")
        if subsets:
            for sub in subsets:
                raw = sub.get("image_dir") or sub.get("image_directory")
                if raw:
                    yield Path(raw)
        else:
            raw = ds.get("image_dir") or ds.get("image_directory")
            if raw:
                yield Path(raw)


def dataset_has_cached_latents(
    toml_path: Path,
    *,
    threshold: float = _CACHED_FRACTION_THRESHOLD,
) -> bool:
    """Return ``True`` when ≥ ``threshold`` fraction of images in the
    dataset have a matching ``.npz`` neighbour.

    A directory with no images is skipped (not counted as cached, not
    counted as uncached). A directory that can't be read is treated as
    uncached and logged at WARNING. The dataset overall must clear the
    threshold across the *union* of all image directories.
    """
    total_images = 0
    cached_images = 0
    seen_any_dir = False
    try:
        dirs = list(_iter_image_dirs(Path(toml_path)))
    except Exception:
        logger.exception("could not parse %s for cache detection", toml_path)
        return False

    for image_dir in dirs:
        try:
            entries = list(image_dir.iterdir())
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            logger.warning(
                "latent-cache detect: cannot read %s; treating as uncached", image_dir
            )
            continue
        seen_any_dir = True

        # Pre-collect .npz stems once per directory so we can do an O(1)
        # membership test per image. sd-scripts names caches
        # ``<image_stem>_<W>x<H>_<arch>.npz``, so the .npz stem starts
        # with the source image's stem followed by ``_``.
        npz_stems = {
            entry.stem
            for entry in entries
            if entry.is_file() and entry.suffix.lower() == ".npz"
        }
        if not npz_stems:
            images_here = sum(
                1 for e in entries
                if e.is_file() and e.suffix.lower() in _IMAGE_SUFFIXES
            )
            total_images += images_here
            continue

        for entry in entries:
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            total_images += 1
            stem = entry.stem
            # Match either exact stem (legacy) or sd-scripts' suffixed
            # form. ``any(...)`` is fine here — npz_stems is bounded by
            # the dir size and the bucket-suffix expansion is small.
            if stem in npz_stems or any(s.startswith(f"{stem}_") for s in npz_stems):
                cached_images += 1

    if not seen_any_dir or total_images == 0:
        return False
    fraction = cached_images / total_images
    logger.info(
        "latent-cache detect: %d/%d images cached (%.1f%%) across %s",
        cached_images,
        total_images,
        fraction * 100.0,
        toml_path,
    )
    return fraction >= threshold
