"""Tests for dataset validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import toml

from bracket.dataset import validator


@pytest.fixture
def temp_dataset_dir():
    """Create a temp directory with a valid image + caption structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        
        # Create subdirectories for different datasets
        for dataset_name in ("good", "empty_caption", "missing_caption", "no_images"):
            dataset_dir = base / dataset_name
            dataset_dir.mkdir(exist_ok=True)
        
        # Good dataset: images with non-empty captions
        good_dir = base / "good"
        (good_dir / "image1.jpg").write_text("dummy image", encoding="utf-8")
        (good_dir / "image1.txt").write_text("a professional portrait", encoding="utf-8")
        (good_dir / "image2.png").write_text("dummy image", encoding="utf-8")
        (good_dir / "image2.txt").write_text("a studio photo", encoding="utf-8")
        
        # Empty caption dataset
        empty_dir = base / "empty_caption"
        (empty_dir / "image3.jpg").write_text("dummy image", encoding="utf-8")
        (empty_dir / "image3.txt").write_text("", encoding="utf-8")  # Empty!
        
        # Missing caption dataset
        missing_dir = base / "missing_caption"
        (missing_dir / "image4.png").write_text("dummy image", encoding="utf-8")
        # No .txt file
        
        # Empty dataset (no images)
        # no_images_dir is already created but empty
        
        yield base


def test_validate_good_dataset(temp_dataset_dir):
    """Test that a valid dataset passes validation."""
    dataset_toml = temp_dataset_dir / "dataset.toml"
    toml.dump({
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(temp_dataset_dir / "good"),
                "num_repeats": 1,
            }
        ]
    }, open(dataset_toml, "w"))
    
    result = validator.validate_dataset_toml(dataset_toml)
    assert result.is_valid
    assert len(result.errors) == 0
    assert "passed" in result.summary.lower()


def test_validate_empty_caption(temp_dataset_dir):
    """Test that empty captions are caught."""
    dataset_toml = temp_dataset_dir / "dataset.toml"
    toml.dump({
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(temp_dataset_dir / "empty_caption"),
                "num_repeats": 1,
            }
        ]
    }, open(dataset_toml, "w"))
    
    result = validator.validate_dataset_toml(dataset_toml)
    assert not result.is_valid
    assert len(result.errors) > 0
    assert any("empty" in e.issue.lower() for e in result.errors)


def test_validate_missing_caption(temp_dataset_dir):
    """Test that missing captions are caught."""
    dataset_toml = temp_dataset_dir / "dataset.toml"
    toml.dump({
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(temp_dataset_dir / "missing_caption"),
                "num_repeats": 1,
            }
        ]
    }, open(dataset_toml, "w"))
    
    result = validator.validate_dataset_toml(dataset_toml)
    assert not result.is_valid
    assert len(result.errors) > 0
    assert any("missing" in e.issue.lower() for e in result.errors)


def test_validate_no_images(temp_dataset_dir):
    """Test that empty directories are treated as warnings."""
    dataset_toml = temp_dataset_dir / "dataset.toml"
    toml.dump({
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(temp_dataset_dir / "no_images"),
                "num_repeats": 1,
            }
        ]
    }, open(dataset_toml, "w"))
    
    result = validator.validate_dataset_toml(dataset_toml)
    # Empty directory is a warning, not a critical error
    assert result.is_valid  # No critical errors
    assert any("no image" in e.issue.lower() for e in result.errors)


def test_fix_empty_captions(temp_dataset_dir):
    """Test auto-fixing empty captions."""
    dataset_toml = temp_dataset_dir / "dataset.toml"
    toml.dump({
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(temp_dataset_dir / "empty_caption"),
                "num_repeats": 1,
            }
        ]
    }, open(dataset_toml, "w"))
    
    # Before fix: validation should fail
    result_before = validator.validate_dataset_toml(dataset_toml)
    assert not result_before.is_valid
    
    # Apply fix
    fixed = validator.fix_empty_captions(dataset_toml, default_caption="test caption")
    assert fixed > 0
    
    # After fix: validation should pass
    result_after = validator.validate_dataset_toml(dataset_toml)
    assert result_after.is_valid


def test_remove_images_without_captions(temp_dataset_dir):
    """Test removing images with missing/empty captions."""
    dataset_toml = temp_dataset_dir / "dataset.toml"
    
    # Dataset with good images + bad images (empty/missing captions)
    toml.dump({
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "image_directory": str(temp_dataset_dir / "good"),
                "num_repeats": 1,
            },
            {
                "image_directory": str(temp_dataset_dir / "empty_caption"),
                "num_repeats": 1,
            },
            {
                "image_directory": str(temp_dataset_dir / "missing_caption"),
                "num_repeats": 1,
            }
        ]
    }, open(dataset_toml, "w"))
    
    # Before removal: should have errors
    result_before = validator.validate_dataset_toml(dataset_toml)
    assert not result_before.is_valid
    
    # Remove problem files
    deleted = validator.remove_images_without_captions(dataset_toml)
    assert deleted > 0
    
    # After removal: validation should pass (good dataset still intact)
    result_after = validator.validate_dataset_toml(dataset_toml)
    assert result_after.is_valid
