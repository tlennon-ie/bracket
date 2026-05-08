"""Per-run TOML derivation + format conversion.

Two dataset-config dialects in the wild:

  sd-scripts (nested):                 musubi-tuner (flat):
    [[datasets]]                          [[datasets]]
    resolution = [1024, 1024]             image_directory = "..."
    batch_size = 1                        cache_directory = "..."
      [[datasets.subsets]]                num_repeats = 1
      image_dir = "..."
      num_repeats = 1

`build_subset()` emits the nested form (sd-scripts-native). Musubi-tuner
scripts (zimage_cache_latents, flux_2_train_network, etc.) reject the nested
form with `extra keys not allowed @ data['datasets'][0]['subsets']`. So
musubi-using adapters call `derive_run_toml(..., target_format="musubi")`
which both injects `batch_size` AND converts the schema.

SDXL adapters use `target_format="sd-scripts"` (the default) which only
injects batch_size — the source is already in the right shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import toml


# ─────────────────────────────── conversion ───────────────────────────────


def _to_musubi_flat(data: dict) -> dict:
    """Transform sd-scripts-nested → musubi-flat.

    Each dataset+subset combination becomes one musubi `[[datasets]]` block
    with `image_directory` (note: NOT `image_dir`) and `num_repeats`.
    Dataset-level keys (resolution, bucket settings) are dropped here because
    musubi reads them from the [general] section instead.
    """
    out_general = dict(data.get("general", {}))
    # Hoist dataset-level config that musubi expects in [general]
    out_datasets: list[dict] = []
    for ds in data.get("datasets", []):
        # Promote dataset-level keys into general (last write wins).
        for hoist_key in ("resolution", "batch_size", "enable_bucket",
                          "bucket_no_upscale", "min_bucket_reso",
                          "max_bucket_reso", "bucket_reso_steps"):
            if hoist_key in ds:
                out_general.setdefault(hoist_key, ds[hoist_key])
        if "subsets" in ds:
            for sub in ds["subsets"]:
                if "image_dir" not in sub and "image_directory" not in sub:
                    continue
                image_dir = sub.get("image_dir") or sub.get("image_directory")
                flat = {
                    "image_directory": image_dir,
                    "num_repeats": int(sub.get("num_repeats", 1)),
                }
                if "caption_extension" in sub:
                    flat["caption_extension"] = sub["caption_extension"]
                out_datasets.append(flat)
        else:
            # Already flat; pass through (rename image_dir → image_directory if present)
            flat = dict(ds)
            if "image_dir" in flat and "image_directory" not in flat:
                flat["image_directory"] = flat.pop("image_dir")
            out_datasets.append(flat)
    return {"general": out_general, "datasets": out_datasets}


def _inject_batch_size_nested(data: dict, batch_size: int) -> None:
    """sd-scripts-nested: batch_size lives at the [[datasets]] level."""
    for ds in data.get("datasets", []):
        ds["batch_size"] = int(batch_size)


def _inject_batch_size_flat(data: dict, batch_size: int) -> None:
    """musubi-flat: batch_size lives in [general]."""
    data.setdefault("general", {})["batch_size"] = int(batch_size)


# ─────────────────────────────── public API ───────────────────────────────


def derive_run_toml(
    source_toml: Path,
    target_path: Path,
    *,
    batch_size: Optional[int] = None,
    target_format: str = "sd-scripts",
) -> Path:
    """Read source TOML, emit per-run TOML in the requested format.

    Args:
        source_toml: subset TOML produced by build_subset (sd-scripts-nested).
        target_path: where to write the derived TOML.
        batch_size: if given, override every dataset's batch_size.
        target_format:
            "sd-scripts" — pass-through (default), inject batch_size at [[datasets]]
            "musubi"     — convert to flat schema, inject batch_size in [general]
    """
    if target_format not in ("sd-scripts", "musubi"):
        raise ValueError(f"target_format must be 'sd-scripts' or 'musubi', got {target_format!r}")

    data = toml.load(Path(source_toml))
    if target_format == "musubi":
        data = _to_musubi_flat(data)
        if batch_size is not None:
            _inject_batch_size_flat(data, batch_size)
    else:
        if batch_size is not None:
            _inject_batch_size_nested(data, batch_size)

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        toml.dump(data, f)
    return target_path
