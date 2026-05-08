"""Tests for derive_run_toml — format conversion + batch_size injection."""
from __future__ import annotations

from pathlib import Path

import pytest
import toml

from bracket.dataset.runtime import derive_run_toml


def _write_nested(path: Path) -> None:
    """sd-scripts nested format — what build_subset emits."""
    path.write_text(toml.dumps({
        "general": {"caption_extension": ".txt"},
        "datasets": [{
            "resolution": [1024, 1024],
            "batch_size": 1,
            "enable_bucket": True,
            "subsets": [
                {"image_dir": "C:/imgs/a", "num_repeats": 4, "caption_extension": ".txt"},
                {"image_dir": "C:/imgs/b", "num_repeats": 16},
            ],
        }],
    }), encoding="utf-8")


def test_sd_scripts_format_passthrough_with_batch_override(tmp_path: Path):
    src = tmp_path / "src.toml"; _write_nested(src)
    out = derive_run_toml(src, tmp_path / "out.toml", batch_size=4)
    data = toml.load(out)
    # Still nested
    assert "subsets" in data["datasets"][0]
    # batch_size at [[datasets]] level
    assert data["datasets"][0]["batch_size"] == 4


def test_musubi_format_converts_subsets_to_flat(tmp_path: Path):
    src = tmp_path / "src.toml"; _write_nested(src)
    out = derive_run_toml(src, tmp_path / "out.toml", batch_size=2, target_format="musubi")
    data = toml.load(out)
    # No nested 'subsets' key
    assert all("subsets" not in ds for ds in data["datasets"])
    # Each [[datasets]] uses image_directory (musubi spelling)
    assert all("image_directory" in ds for ds in data["datasets"])
    assert all("image_dir" not in ds for ds in data["datasets"])
    # 2 datasets (one per source subset)
    assert len(data["datasets"]) == 2
    assert data["datasets"][0]["num_repeats"] == 4
    assert data["datasets"][1]["num_repeats"] == 16
    # batch_size hoisted to [general]
    assert data["general"]["batch_size"] == 2
    # Resolution + enable_bucket also hoisted
    assert data["general"]["resolution"] == [1024, 1024]
    assert data["general"]["enable_bucket"] is True


def test_musubi_format_without_batch_size_omits_general_override(tmp_path: Path):
    src = tmp_path / "src.toml"; _write_nested(src)
    out = derive_run_toml(src, tmp_path / "out.toml", target_format="musubi")
    data = toml.load(out)
    # Source-level batch_size=1 was hoisted into general
    assert data["general"]["batch_size"] == 1


def test_invalid_target_format_raises(tmp_path: Path):
    src = tmp_path / "src.toml"; _write_nested(src)
    with pytest.raises(ValueError, match="target_format"):
        derive_run_toml(src, tmp_path / "out.toml", target_format="exotic")


def test_already_flat_input_passes_through_for_musubi(tmp_path: Path):
    """If a user feeds a flat TOML directly, musubi conversion is idempotent."""
    src = tmp_path / "src.toml"
    src.write_text(toml.dumps({
        "general": {"caption_extension": ".txt", "resolution": [1024, 1024]},
        "datasets": [
            {"image_directory": "C:/x", "num_repeats": 4},
        ],
    }), encoding="utf-8")
    out = derive_run_toml(src, tmp_path / "out.toml", batch_size=8, target_format="musubi")
    data = toml.load(out)
    assert data["datasets"][0]["image_directory"] == "C:/x"
    assert data["general"]["batch_size"] == 8
