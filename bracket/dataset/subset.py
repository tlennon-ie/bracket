"""Build a small dataset subset that completes a candidate run in minutes.

Reads a flat musubi-tuner-style TOML (one `[[datasets]]` per image directory
with `image_directory`/`cache_directory`/`num_repeats` keys), samples K images
per directory, and writes a derived TOML in **sd-scripts native format**:

    [general]
    caption_extension = ".txt"

    [[datasets]]
    resolution = [1024, 1024]
    batch_size = 1
    enable_bucket = true
    bucket_no_upscale = false

      [[datasets.subsets]]
      image_dir = "..."
      num_repeats = 4

This format is what sd-scripts' voluptuous schema accepts. We translate at
generation time so the orchestrator works with both source formats.
"""

from __future__ import annotations

import random as _random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import toml


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# Keys that live at the [[datasets]] level in sd-scripts (passed through if
# present in the input [general] or [[datasets]] sections).
_DATASET_LEVEL_KEYS = {
    "resolution",
    "batch_size",
    "enable_bucket",
    "bucket_no_upscale",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "network_multiplier",
}

# Keys allowed on a sd-scripts subset (DreamBooth flavor — what we use for
# image+caption directories).
_SUBSET_LEVEL_KEYS = {
    "image_dir",
    "num_repeats",
    "caption_extension",
    "class_tokens",
    "is_reg",
    "keep_tokens",
    "color_aug",
    "flip_aug",
    "random_crop",
    "shuffle_caption",
    "caption_dropout_rate",
    "caption_dropout_every_n_epochs",
    "caption_tag_dropout_rate",
    "caption_prefix",
    "caption_suffix",
}


@dataclass(frozen=True)
class DatasetSubsetSpec:
    images_per_dataset: int = 24
    caption_extension: str = ".txt"
    seed: int = 0
    copy_files: bool = True


def _sample_images(
    src_dir: Path, k: int, exts: set[str], rng: _random.Random
) -> list[Path]:
    candidates = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not candidates:
        return []
    if len(candidates) <= k:
        return candidates
    return rng.sample(candidates, k)


def build_subset(
    source_toml: Path,
    target_dir: Path,
    *,
    spec: Optional[DatasetSubsetSpec] = None,
) -> Path:
    spec = spec or DatasetSubsetSpec()
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = _random.Random(spec.seed)

    src_data = toml.load(source_toml)
    src_general = dict(src_data.get("general", {}))
    caption_ext = src_general.get("caption_extension", spec.caption_extension)

    # Collect dataset-level options. The musubi-tuner-flavored TOML sometimes
    # puts these in [general] (since each [[datasets]] inherits them); promote
    # them into the [[datasets]] block sd-scripts expects.
    dataset_level: dict = {}
    for k in _DATASET_LEVEL_KEYS:
        if k in src_general:
            dataset_level[k] = src_general[k]
    # Sensible defaults if the source didn't supply them
    dataset_level.setdefault("resolution", [1024, 1024])
    dataset_level.setdefault("batch_size", 1)
    dataset_level.setdefault("enable_bucket", True)
    dataset_level.setdefault("bucket_no_upscale", False)

    new_subsets: list[dict] = []
    for i, ds in enumerate(src_data.get("datasets", [])):
        # Source can be either flat-musubi (image_directory) or already-sd-scripts (subsets)
        candidate_dirs: list[tuple[Path, int, dict]] = []
        if "image_directory" in ds:
            candidate_dirs.append(
                (Path(ds["image_directory"]), int(ds.get("num_repeats", 1)), ds)
            )
        elif "subsets" in ds:
            for sub in ds["subsets"]:
                if "image_dir" in sub:
                    candidate_dirs.append(
                        (Path(sub["image_dir"]), int(sub.get("num_repeats", 1)), sub)
                    )
                elif "image_directory" in sub:
                    candidate_dirs.append(
                        (Path(sub["image_directory"]), int(sub.get("num_repeats", 1)), sub)
                    )

        for j, (src_images, num_repeats, src_keys) in enumerate(candidate_dirs):
            if not src_images.exists():
                raise FileNotFoundError(f"image directory does not exist: {src_images}")
            sampled = _sample_images(src_images, spec.images_per_dataset, _IMAGE_EXTS, rng)
            if not sampled:
                continue
            slug = f"d{i:02d}_{j:02d}_{src_images.name}"
            if spec.copy_files:
                dest = target_dir / slug
                dest.mkdir(parents=True, exist_ok=True)
                for img in sampled:
                    shutil.copy2(img, dest / img.name)
                    cap = img.with_suffix(caption_ext)
                    if cap.exists():
                        shutil.copy2(cap, dest / cap.name)
                image_dir = dest
            else:
                image_dir = src_images
            subset_entry: dict = {
                "image_dir": str(image_dir).replace("\\", "/"),
                "num_repeats": num_repeats,
                "caption_extension": caption_ext,
            }
            # Forward any other subset-level keys the user set.
            for k, v in src_keys.items():
                if k in _SUBSET_LEVEL_KEYS and k not in subset_entry:
                    subset_entry[k] = v
            new_subsets.append(subset_entry)

    derived = {
        "general": {"caption_extension": caption_ext},
        "datasets": [{**dataset_level, "subsets": new_subsets}],
    }
    out_path = target_dir / "dataset.toml"
    with out_path.open("w", encoding="utf-8") as f:
        toml.dump(derived, f)
    return out_path
