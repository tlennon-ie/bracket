"""CLI for dataset validation and repair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bracket.dataset.validator import (
    fix_empty_captions,
    remove_images_without_captions,
    validate_dataset_toml,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and repair dataset TOML files."
    )
    parser.add_argument(
        "dataset_toml",
        type=Path,
        help="Path to dataset.toml file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only validate, don't fix (default)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-populate empty captions with default text",
    )
    parser.add_argument(
        "--remove-bad",
        action="store_true",
        help="Remove images without valid captions",
    )
    parser.add_argument(
        "--default-caption",
        type=str,
        default="professional",
        help="Default caption text for auto-fixing (default: 'professional')",
    )

    args = parser.parse_args()
    dataset_toml = Path(args.dataset_toml).resolve()

    if not dataset_toml.exists():
        print(f"Error: Dataset TOML not found: {dataset_toml}", file=sys.stderr)
        return 1

    # Validate
    result = validate_dataset_toml(dataset_toml)
    print(f"\n{result.summary}\n")

    if result.errors:
        for err in result.errors:
            severity_str = f"[{err.severity.upper()}]"
            print(f"  {severity_str} {err.image_path}")
            print(f"      {err.issue}")

    if not result.is_valid:
        print(
            f"\nDataset has critical errors. Fix with:\n"
            f"  python -m bracket.dataset.validator_cli --fix {dataset_toml}\n"
            f"  python -m bracket.dataset.validator_cli --remove-bad {dataset_toml}\n",
            file=sys.stderr,
        )
        if args.fix or args.remove_bad:
            # User explicitly asked to fix, so proceed
            pass
        else:
            return 1

    # Apply fixes if requested
    if args.fix:
        print(f"Auto-populating empty captions with: '{args.default_caption}'")
        fixed = fix_empty_captions(dataset_toml, default_caption=args.default_caption)
        print(f"Fixed {fixed} file(s).\n")

        # Re-validate to confirm
        result = validate_dataset_toml(dataset_toml)
        print(f"After fix: {result.summary}\n")
        if not result.is_valid:
            print("Still has errors. Manual intervention needed.", file=sys.stderr)
            return 1
        return 0

    if args.remove_bad:
        print("Removing images without valid captions...")
        deleted = remove_images_without_captions(dataset_toml)
        print(f"Deleted {deleted} file(s) (images + captions).\n")

        # Re-validate to confirm
        result = validate_dataset_toml(dataset_toml)
        print(f"After removal: {result.summary}\n")
        if not result.is_valid:
            print("Still has errors. Manual intervention needed.", file=sys.stderr)
            return 1
        return 0

    # Default: just check
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
