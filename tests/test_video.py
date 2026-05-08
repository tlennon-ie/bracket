"""Tests for `bracket.video` — frame extraction utilities for video samples.

Frame extraction itself uses ffmpeg, which we don't depend on for tests.
These cases lock the lightweight helpers (extension detection, listing).
"""
from __future__ import annotations

from pathlib import Path

from bracket.video import (
    VIDEO_EXTENSIONS,
    extract_frames,
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
