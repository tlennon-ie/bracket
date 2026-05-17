"""Dataset validation — catch common issues before training starts.

Validates:
  - Caption files exist for every image
  - Caption files are non-empty
  - Image directories exist and have images
  - Resolves issues with actionable error messages
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import toml


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ValidationError(NamedTuple):
    image_path: Path
    issue: str
    severity: str  # "error" | "warning"


class ValidationResult(NamedTuple):
    is_valid: bool
    errors: list[ValidationError]
    summary: str


def validate_dataset_toml(dataset_toml: Path) -> ValidationResult:
    """Validate a dataset TOML before training.

    Returns:
        ValidationResult with is_valid=True only if all critical checks pass.
    """
    if not dataset_toml.exists():
        return ValidationResult(
            is_valid=False,
            errors=[],
            summary=f"Dataset TOML does not exist: {dataset_toml}",
        )

    try:
        data = toml.load(dataset_toml)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            errors=[],
            summary=f"Failed to parse dataset TOML: {e}",
        )

    errors: list[ValidationError] = []

    # Determine caption extension
    caption_ext = data.get("general", {}).get("caption_extension", ".txt")
    if not caption_ext.startswith("."):
        caption_ext = f".{caption_ext}"

    # Validate each dataset + subset
    for ds_idx, ds in enumerate(data.get("datasets", [])):
        candidate_dirs: list[tuple[Path, int]] = []

        if "image_directory" in ds:
            candidate_dirs.append((Path(ds["image_directory"]), int(ds.get("num_repeats", 1))))
        elif "subsets" in ds:
            for sub in ds["subsets"]:
                if "image_dir" in sub:
                    candidate_dirs.append((Path(sub["image_dir"]), int(sub.get("num_repeats", 1))))
                elif "image_directory" in sub:
                    candidate_dirs.append(
                        (Path(sub["image_directory"]), int(sub.get("num_repeats", 1)))
                    )

        for img_dir, num_repeats in candidate_dirs:
            if not img_dir.exists():
                errors.append(
                    ValidationError(
                        image_path=img_dir,
                        issue=f"Image directory does not exist",
                        severity="error",
                    )
                )
                continue

            # Find all images in this directory
            images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
            if not images:
                errors.append(
                    ValidationError(
                        image_path=img_dir,
                        issue="No image files found in directory (skipping)",
                        severity="warning",
                    )
                )
                continue

            # Check each image for a non-empty caption
            for img_path in images:
                cap_path = img_path.with_suffix(caption_ext)
                if not cap_path.exists():
                    errors.append(
                        ValidationError(
                            image_path=img_path,
                            issue=f"Caption file missing: {cap_path.name}",
                            severity="error",
                        )
                    )
                else:
                    # Check if caption is empty
                    try:
                        content = cap_path.read_text(encoding="utf-8").strip()
                        if not content:
                            errors.append(
                                ValidationError(
                                    image_path=img_path,
                                    issue=f"Caption file is empty: {cap_path.name}",
                                    severity="error",
                                )
                            )
                    except Exception as e:
                        errors.append(
                            ValidationError(
                                image_path=img_path,
                                issue=f"Failed to read caption: {e}",
                                severity="warning",
                            )
                        )

    # Summarize
    critical = [e for e in errors if e.severity == "error"]
    warnings = [e for e in errors if e.severity == "warning"]

    if critical:
        summary = f"{len(critical)} critical issue(s) found, {len(warnings)} warning(s)."
        is_valid = False
    elif warnings:
        summary = f"No critical issues, but {len(warnings)} warning(s)."
        is_valid = True
    else:
        summary = "Dataset validation passed."
        is_valid = True

    return ValidationResult(is_valid=is_valid, errors=errors, summary=summary)


def fix_empty_captions(dataset_toml: Path, default_caption: str = "professional") -> int:
    """Auto-populate empty caption files with a default string.

    Returns the number of files fixed.
    """
    if not dataset_toml.exists():
        raise FileNotFoundError(f"Dataset TOML does not exist: {dataset_toml}")

    data = toml.load(dataset_toml)
    caption_ext = data.get("general", {}).get("caption_extension", ".txt")
    if not caption_ext.startswith("."):
        caption_ext = f".{caption_ext}"

    fixed_count = 0

    for ds in data.get("datasets", []):
        candidate_dirs: list[Path] = []

        if "image_directory" in ds:
            candidate_dirs.append(Path(ds["image_directory"]))
        elif "subsets" in ds:
            for sub in ds["subsets"]:
                if "image_dir" in sub:
                    candidate_dirs.append(Path(sub["image_dir"]))
                elif "image_directory" in sub:
                    candidate_dirs.append(Path(sub["image_directory"]))

        for img_dir in candidate_dirs:
            if not img_dir.exists():
                continue

            images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
            for img_path in images:
                cap_path = img_path.with_suffix(caption_ext)
                if cap_path.exists():
                    try:
                        content = cap_path.read_text(encoding="utf-8").strip()
                        if not content:
                            cap_path.write_text(default_caption, encoding="utf-8")
                            fixed_count += 1
                    except Exception:
                        pass
                else:
                    # Create missing caption with default
                    cap_path.write_text(default_caption, encoding="utf-8")
                    fixed_count += 1

    return fixed_count


def remove_images_without_captions(dataset_toml: Path) -> int:
    """Remove images that don't have valid non-empty captions.

    Returns the number of files deleted.
    """
    if not dataset_toml.exists():
        raise FileNotFoundError(f"Dataset TOML does not exist: {dataset_toml}")

    data = toml.load(dataset_toml)
    caption_ext = data.get("general", {}).get("caption_extension", ".txt")
    if not caption_ext.startswith("."):
        caption_ext = f".{caption_ext}"

    deleted_count = 0

    for ds in data.get("datasets", []):
        candidate_dirs: list[Path] = []

        if "image_directory" in ds:
            candidate_dirs.append(Path(ds["image_directory"]))
        elif "subsets" in ds:
            for sub in ds["subsets"]:
                if "image_dir" in sub:
                    candidate_dirs.append(Path(sub["image_dir"]))
                elif "image_directory" in sub:
                    candidate_dirs.append(Path(sub["image_directory"]))

        for img_dir in candidate_dirs:
            if not img_dir.exists():
                continue

            images = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
            for img_path in images:
                cap_path = img_path.with_suffix(caption_ext)
                
                # Mark for deletion if: (1) caption doesn't exist, or (2) caption is empty
                should_delete = False
                if not cap_path.exists():
                    should_delete = True
                else:
                    try:
                        content = cap_path.read_text(encoding="utf-8").strip()
                        if not content:
                            should_delete = True
                    except Exception:
                        # If we can't read it, delete both
                        should_delete = True

                if should_delete:
                    img_path.unlink()
                    if cap_path.exists():
                        cap_path.unlink()
                    deleted_count += 1

    return deleted_count
