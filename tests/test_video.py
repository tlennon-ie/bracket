"""Tests for `bracket.video` — frame extraction utilities for video samples.

Frame extraction itself uses ffmpeg, which we don't depend on for tests.
These cases lock the lightweight helpers (extension detection, listing).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from bracket.video import (
    ANIMATED_IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    extract_frames,
    is_animated_image,
    is_frame_extractable,
    is_video_file,
    list_video_samples,
)


def test_video_extensions_cover_common_formats():
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".webm" in VIDEO_EXTENSIONS
    assert ".mov" in VIDEO_EXTENSIONS
    assert ".mkv" in VIDEO_EXTENSIONS


def test_is_video_file_extensions(tmp_path):
    for name in ("clip.mp4", "clip.MP4", "clip.webm", "movie.mov"):
        assert is_video_file(tmp_path / name) is True
    for name in ("frame.png", "report.md", "weights.safetensors"):
        assert is_video_file(tmp_path / name) is False


def test_list_video_samples_skips_non_video(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "c.webm").write_bytes(b"")
    out = list_video_samples(tmp_path)
    names = sorted(p.name for p in out)
    assert names == ["a.mp4", "c.webm"]


def test_list_video_samples_for_missing_dir(tmp_path):
    assert list_video_samples(tmp_path / "nope") == []


def test_extract_frames_missing_video_returns_error(tmp_path):
    res = extract_frames(tmp_path / "missing.mp4", out_dir=tmp_path / "frames")
    assert res.frames == ()
    assert res.error == "video_missing"


# ─────────────── animated WebP/GIF (ai-toolkit video samples) ───────────────
#
# ai-toolkit forces `output_ext = webp` for any sample with num_frames > 1, so
# its LTX-2.5 / MiniMax-H3 runs emit ANIMATED WebP where every other video
# trainer emits .mp4. `.webp` is also a normal still extension, so a purely
# extension-based check judges the whole clip on one frame and silently scores
# a video model as if it were a stills model.


def _write_webp(path: Path, *, frames: int, size=(16, 16)) -> Path:
    """Write a WebP with `frames` frames (1 = an ordinary still)."""
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    imgs = [
        Image.new("RGB", size, (i * 20 % 256, 40, 80)) for i in range(max(1, frames))
    ]
    if frames > 1:
        imgs[0].save(
            path, format="WEBP", save_all=True, append_images=imgs[1:], duration=100,
        )
    else:
        imgs[0].save(path, format="WEBP")
    return path


def test_animated_extensions_are_the_ambiguous_ones():
    # Both can hold a still OR an animation — which is the whole problem.
    assert ".webp" in ANIMATED_IMAGE_EXTENSIONS
    assert ".gif" in ANIMATED_IMAGE_EXTENSIONS
    # ...and neither is a plain video container.
    for ext in ANIMATED_IMAGE_EXTENSIONS:
        assert ext not in VIDEO_EXTENSIONS


def test_is_animated_image_distinguishes_by_content_not_extension(tmp_path):
    animated = _write_webp(tmp_path / "anim.webp", frames=5)
    still = _write_webp(tmp_path / "still.webp", frames=1)
    assert is_animated_image(animated) is True
    assert is_animated_image(still) is False
    # Extensions that are never animated short-circuit without a file read.
    assert is_animated_image(tmp_path / "frame.png") is False


def test_is_animated_image_tolerates_unreadable_file(tmp_path):
    """A truncated sample must degrade to 'still', never raise."""
    junk = tmp_path / "truncated.webp"
    junk.write_bytes(b"not really a webp")
    assert is_animated_image(junk) is False


def test_is_frame_extractable_spans_both_families(tmp_path):
    animated = _write_webp(tmp_path / "anim.webp", frames=4)
    still = _write_webp(tmp_path / "still.webp", frames=1)
    assert is_frame_extractable(animated) is True
    assert is_frame_extractable(tmp_path / "clip.mp4") is True
    assert is_frame_extractable(still) is False
    assert is_frame_extractable(tmp_path / "frame.png") is False


def test_list_video_samples_picks_up_animated_webp_only(tmp_path):
    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()
    animated = _write_webp(sample_dir / "anim.webp", frames=6)
    _write_webp(sample_dir / "still.webp", frames=1)
    (sample_dir / "frame.png").write_bytes(b"\x00")
    (sample_dir / "clip.mp4").write_bytes(b"\x00")

    found = list_video_samples(sample_dir)
    assert animated in found
    assert sample_dir / "clip.mp4" in found
    assert sample_dir / "still.webp" not in found
    assert sample_dir / "frame.png" not in found


def test_extract_frames_from_animated_webp_without_ffmpeg(tmp_path):
    """The Pillow path must work with no ffmpeg on PATH (ffprobe can't do WebP)."""
    animated = _write_webp(tmp_path / "anim.webp", frames=9)
    result = extract_frames(animated, out_dir=tmp_path / "_frames", n_frames=3)
    assert result.error is None
    assert len(result.frames) == 3
    for f in result.frames:
        assert f.exists() and f.stat().st_size > 0
        assert f.suffix == ".png"
    # Stems must match the ffmpeg backend's so prompt pairing is unchanged.
    assert [f.name for f in result.frames] == [
        "anim_frame00.png", "anim_frame01.png", "anim_frame02.png",
    ]


def test_extract_frames_clamps_to_available_frame_count(tmp_path):
    """Asking for more frames than the clip has must not duplicate or crash."""
    animated = _write_webp(tmp_path / "short.webp", frames=2)
    result = extract_frames(animated, out_dir=tmp_path / "_frames", n_frames=3)
    assert result.error is None
    assert len(result.frames) == 2
