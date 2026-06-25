"""ai-toolkit dataset bridge — Bracket dataset.toml → folder_path + caption_ext.

ostris' ``ai-toolkit`` (the ``sd_trainer`` job) does not consume Bracket's
sd-scripts/musubi-tuner ``dataset.toml``. Its ``datasets`` config block points
directly at an **image folder** (``folder_path``) plus a caption extension
(``caption_ext``); captions are sidecar text files with the same stem as each
image. There is no separate manifest file to write.

This module is the one-way bridge that extracts those two values from the
source dataset TOML (the sd-scripts-nested shape Bracket writes —
``[[datasets]]`` → ``[[datasets.subsets]]`` with an ``image_dir`` and an
optional ``caption_extension``; or a flat ``image_directory``). Only the FIRST
declared media directory is used: ai-toolkit's ``datasets`` block is one folder
per entry, and Bracket sessions train on a single subject directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import toml

logger = logging.getLogger("bracket.dataset.aitk_dataset")

# ai-toolkit's caption_ext is the bare extension WITHOUT a leading dot
# (e.g. "txt", not ".txt"). Bracket's TOML carries it with the dot, so we
# normalise on the way out.
_DEFAULT_CAPTION_EXT = "txt"


def _normalise_caption_ext(ext: str) -> str:
    """Return the caption extension in ai-toolkit's dot-less form."""
    if not isinstance(ext, str) or not ext:
        return _DEFAULT_CAPTION_EXT
    return ext.lstrip(".") or _DEFAULT_CAPTION_EXT


def _general_caption_ext(data: dict) -> str:
    """Resolve the [general]-level caption extension (dot-less)."""
    return _normalise_caption_ext(
        data.get("general", {}).get("caption_extension", _DEFAULT_CAPTION_EXT)
    )


def image_dir_for(source_toml: Path) -> tuple[str, str]:
    """Extract the ai-toolkit ``(folder_path, caption_ext)`` from a dataset TOML.

    Reads the source ``dataset.toml`` and returns the first declared media
    directory as an absolute ``folder_path`` string, paired with the caption
    extension in ai-toolkit's dot-less form (subset-level
    ``caption_extension`` overrides the [general] default).

    Args:
        source_toml: sd-scripts/musubi dataset TOML.

    Returns:
        ``(folder_path, caption_ext)`` — ``folder_path`` is an absolute path
        string; ``caption_ext`` is e.g. ``"txt"``.

    Raises:
        FileNotFoundError: when ``source_toml`` does not exist.
        ValueError: when no media directory can be found in the TOML.
    """
    source_toml = Path(source_toml)
    if not source_toml.exists():
        raise FileNotFoundError(f"source dataset TOML does not exist: {source_toml}")

    data = toml.load(source_toml)
    general_ext = _general_caption_ext(data)

    for ds in data.get("datasets", []):
        if "image_directory" in ds:
            ext = _normalise_caption_ext(str(ds.get("caption_extension", general_ext)))
            return str(Path(ds["image_directory"]).resolve()), ext
        for sub in ds.get("subsets", []):
            media = sub.get("image_dir") or sub.get("image_directory")
            if not media:
                continue
            ext = _normalise_caption_ext(str(sub.get("caption_extension", general_ext)))
            return str(Path(media).resolve()), ext

    raise ValueError(
        f"no image directory found in dataset TOML: {source_toml} "
        "(expected [[datasets.subsets]].image_dir or [[datasets]].image_directory)"
    )
