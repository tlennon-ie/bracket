"""LTX-2 dataset bridge — sd-scripts/musubi TOML → ltx-trainer dataset.json.

Lightricks' native ``ltx-trainer`` (LTX-2) does not consume the sd-scripts /
musubi-tuner ``dataset.toml`` dialects. Its ``scripts/process_dataset.py``
preprocessor reads a JSON list whose recognized columns are ``video`` (path to
the media file) and ``caption`` (text), e.g.::

    [{"video": "/abs/clip1.mp4", "caption": "a cat"}, ...]

This module is the one-way bridge:

  * :func:`build_ltx2_dataset_json` reads the source ``dataset.toml`` (the
    sd-scripts-nested shape Bracket writes — ``[[datasets]]`` →
    ``[[datasets.subsets]]`` with an ``image_dir`` and optional
    ``caption_extension``), enumerates the media files, pairs each with its
    sidecar caption (same stem + caption_extension, default ``.txt``; empty
    string when absent), and writes the JSON list.
  * :func:`preprocessed_root_for` returns the DETERMINISTIC directory where
    ``process_dataset.py`` writes ``latents/`` / ``conditions/`` /
    ``audio_latents/``. It is derived from the source dataset location (NOT a
    per-candidate run_dir) so the single, expensive preprocessing pass is
    reused by every candidate in a session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import toml

logger = logging.getLogger("bracket.dataset.ltx2_dataset")


# Media extensions LTX-2 treats as video clips. process_dataset.py reads the
# whole clip, so we enumerate any of these as a ``video`` column entry.
_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")

# Deterministic preprocessing-root directory name, placed next to the source
# dataset.toml. Must match between session_setup_commands (which writes it)
# and prepare_run (which references it).
_PREPROCESSED_DIRNAME = ".precomputed_ltx2"


def preprocessed_root_for(source_toml: Path) -> Path:
    """Return the deterministic LTX-2 preprocessed-data root for a dataset.

    Derived from the source ``dataset.toml`` location (its parent directory)
    so the one ``process_dataset.py`` pass is reused by every candidate.
    """
    return Path(source_toml).resolve().parent / _PREPROCESSED_DIRNAME


def _caption_extension(data: dict) -> str:
    """Resolve the caption extension from the source TOML ([general] level)."""
    ext = data.get("general", {}).get("caption_extension", ".txt")
    if not isinstance(ext, str) or not ext:
        ext = ".txt"
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def _iter_media_dirs(data: dict) -> list[tuple[Path, str]]:
    """Yield (media_directory, caption_extension) pairs from the source TOML.

    Mirrors ``bracket.dataset.validator`` traversal: a dataset may declare its
    directory directly (``image_directory``) or via nested ``subsets`` with an
    ``image_dir`` / ``image_directory``. A subset-level ``caption_extension``
    overrides the [general] default.
    """
    general_ext = _caption_extension(data)
    out: list[tuple[Path, str]] = []
    for ds in data.get("datasets", []):
        if "image_directory" in ds:
            out.append((Path(ds["image_directory"]), str(ds.get("caption_extension", general_ext))))
            continue
        for sub in ds.get("subsets", []):
            media = sub.get("image_dir") or sub.get("image_directory")
            if not media:
                continue
            ext = str(sub.get("caption_extension", general_ext))
            if not ext.startswith("."):
                ext = f".{ext}"
            out.append((Path(media), ext))
    return out


def _read_caption(media_path: Path, caption_ext: str) -> str:
    """Read the sidecar caption for a media file; empty string if absent."""
    cap_path = media_path.with_suffix(caption_ext)
    if not cap_path.exists():
        return ""
    try:
        return cap_path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("failed to read caption %s; using empty string", cap_path)
        return ""


def build_ltx2_dataset_json(source_toml: Path, target_json: Path) -> Path:
    """Read the source dataset TOML and write an ltx-trainer dataset.json.

    Args:
        source_toml: sd-scripts/musubi dataset TOML (media dir + caption conv).
        target_json: where to write the JSON list.

    Returns:
        The ``target_json`` path (written).

    Raises:
        FileNotFoundError: when ``source_toml`` does not exist.
    """
    source_toml = Path(source_toml)
    if not source_toml.exists():
        raise FileNotFoundError(f"source dataset TOML does not exist: {source_toml}")

    data = toml.load(source_toml)
    rows: list[dict[str, str]] = []
    for media_dir, caption_ext in _iter_media_dirs(data):
        if not media_dir.exists():
            logger.warning("media directory does not exist, skipping: %s", media_dir)
            continue
        clips = sorted(
            p for p in media_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
        )
        for clip in clips:
            rows.append({
                "video": str(clip.resolve()),
                "caption": _read_caption(clip, caption_ext),
            })

    if not rows:
        logger.warning("no media clips found for %s; writing empty dataset.json", source_toml)

    target_json = Path(target_json)
    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("wrote ltx-2 dataset.json with %d clip(s): %s", len(rows), target_json)
    return target_json
