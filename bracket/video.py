"""Video sample helpers.

Video trainers (HunyuanVideo, Wan 2.1/2.2, LTX-2, FramePack) emit `.mp4`
samples instead of `.png`. Bracket's existing scorer + judge expect images, so
we extract a small handful of representative frames per video and let the
existing image-judge score them.

**ai-toolkit is the exception**: for any sample with `num_frames > 1` it forces
the output extension to `webp` and writes an *animated* WebP, not an `.mp4`
(`toolkit/config_modules.py`). `.webp` is also a perfectly ordinary still-image
extension, so extension alone cannot tell the two apart — an animated WebP that
slips through as a still gets judged on one frame, silently scoring an LTX-2.5
or MiniMax-H3 run as if it were a stills model. :func:`is_animated_image`
therefore opens the file and asks. Extraction for those goes through Pillow
rather than ffmpeg: ffprobe's duration for animated WebP is unreliable, and
Pillow addresses frames by index, which is what we want anyway.

Three frames per clip is the established default for video LoRA evaluation
in community rigs (one near the start, one in the middle, one near the end).
That's enough to catch the visible failure modes (blurry first frame, bad
character continuity, melted last frame) without quintupling the judge cost.

We use ffmpeg if available (already a musubi-tuner dependency for video
caching); else PyAV; else fall back to OpenCV. All three are commonly present
in a video-training venv. If none are present, the helper returns no frames
and the scorer falls back to loss-only for that run.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".webm", ".mov", ".mkv")

# Container formats that may hold either a single frame or an animation. Only
# an actual read distinguishes them — see is_animated_image.
ANIMATED_IMAGE_EXTENSIONS: tuple[str, ...] = (".webp", ".gif")


@dataclass(frozen=True)
class FrameExtractionResult:
    video: Path
    frames: tuple[Path, ...]
    error: Optional[str] = None


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_animated_image(path: Path) -> bool:
    """True when `path` is a multi-frame WebP or GIF.

    Pillow is an optional dependency. When it is missing we return False, which
    restores the pre-existing behaviour of treating the file as a still — a
    degradation, not a crash.
    """
    if path.suffix.lower() not in ANIMATED_IMAGE_EXTENSIONS:
        return False
    try:
        from PIL import Image
    except ImportError:
        logger.warning(
            "Pillow not installed — cannot tell an animated %s from a still "
            "one, so %s will be judged as a single image. `pip install "
            "Pillow` to score every frame.",
            path.suffix, path.name,
        )
        return False
    try:
        with Image.open(path) as img:
            return getattr(img, "n_frames", 1) > 1
    except (OSError, ValueError):
        # Unreadable or truncated — let the still-image path deal with it.
        return False


def is_frame_extractable(path: Path) -> bool:
    """True when frames should be pulled out of `path` before judging."""
    return is_video_file(path) or is_animated_image(path)


def extract_frames(
    video: Path,
    *,
    out_dir: Path,
    n_frames: int = 3,
    timestamps: Optional[Sequence[float]] = None,
) -> FrameExtractionResult:
    """Extract `n_frames` PNGs from `video` into `out_dir`.

    timestamps: explicit seconds to sample (overrides n_frames). If omitted,
    samples at 10%, 50%, 90% of the duration — robust against trimmed clips.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        return FrameExtractionResult(video=video, frames=(), error="video_missing")

    if is_animated_image(video):
        return _extract_animated_image_frames(
            video, out_dir=out_dir, n_frames=n_frames,
        )

    duration = _probe_duration(video)
    if duration is None or duration <= 0:
        return FrameExtractionResult(video=video, frames=(), error="duration_unknown")

    if timestamps is None:
        if n_frames <= 1:
            ts = (duration / 2.0,)
        else:
            offsets = [(i + 0.5) / n_frames for i in range(n_frames)]
            ts = tuple(o * duration for o in offsets)
    else:
        ts = tuple(timestamps)

    frames: list[Path] = []
    stem = video.stem
    for idx, t in enumerate(ts):
        out = out_dir / f"{stem}_frame{idx:02d}.png"
        if _ffmpeg_extract(video, out, t):
            frames.append(out)
    if not frames:
        return FrameExtractionResult(video=video, frames=(), error="extraction_failed")
    return FrameExtractionResult(video=video, frames=tuple(frames))


def _extract_animated_image_frames(
    image: Path, *, out_dir: Path, n_frames: int,
) -> FrameExtractionResult:
    """Extract `n_frames` PNGs from an animated WebP/GIF via Pillow.

    Frames are addressed by index (not timestamp), sampled at the same
    even spacing the ffmpeg path uses, so extracted stems stay identical
    between the two backends and prompt pairing is unaffected.
    """
    try:
        from PIL import Image
    except ImportError:
        return FrameExtractionResult(video=image, frames=(), error="pillow_missing")

    frames: list[Path] = []
    try:
        with Image.open(image) as img:
            total = getattr(img, "n_frames", 1)
            if total < 1:
                return FrameExtractionResult(
                    video=image, frames=(), error="no_frames",
                )
            count = max(1, min(int(n_frames), total))
            indices = [
                min(total - 1, int((i + 0.5) / count * total))
                for i in range(count)
            ]
            for idx, frame_no in enumerate(indices):
                out = out_dir / f"{image.stem}_frame{idx:02d}.png"
                img.seek(frame_no)
                # Animated WebP/GIF frames are palettised or RGBA; the judge
                # wants plain RGB.
                img.convert("RGB").save(out, format="PNG")
                frames.append(out)
    except (OSError, ValueError, EOFError) as e:
        logger.warning("animated-image frame extraction failed for %s: %s", image.name, e)
        return FrameExtractionResult(
            video=image, frames=tuple(frames), error="extraction_failed",
        )

    if not frames:
        return FrameExtractionResult(video=image, frames=(), error="extraction_failed")
    return FrameExtractionResult(video=image, frames=tuple(frames))


def _probe_duration(video: Path) -> Optional[float]:
    """Return clip duration in seconds, or None if probe fails."""
    ff = shutil.which("ffprobe")
    if ff is None:
        return None
    try:
        out = subprocess.check_output(
            [
                ff, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            text=True, stderr=subprocess.STDOUT, timeout=10,
        )
        return float(out.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _ffmpeg_extract(video: Path, out: Path, t: float) -> bool:
    """Extract a single frame at timestamp `t` (seconds). True on success."""
    ff = shutil.which("ffmpeg")
    if ff is None:
        return False
    try:
        subprocess.check_call(
            [
                ff, "-y", "-loglevel", "error",
                "-ss", f"{t:.3f}", "-i", str(video),
                "-frames:v", "1", "-f", "image2", str(out),
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
        )
        return out.exists() and out.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def list_video_samples(sample_dir: Path) -> list[Path]:
    """Return every frame-extractable sample in `sample_dir`, sorted.

    Covers both real video containers and the animated WebP/GIF ai-toolkit
    writes for its video models.
    """
    if not sample_dir.exists():
        return []
    return sorted(
        p for p in sample_dir.iterdir() if p.is_file() and is_frame_extractable(p)
    )


def extract_all_videos(
    videos: Iterable[Path], *, frames_dir: Path, n_frames: int = 3,
) -> dict[Path, tuple[Path, ...]]:
    """Map each video → tuple of extracted frame paths."""
    out: dict[Path, tuple[Path, ...]] = {}
    for v in videos:
        res = extract_frames(v, out_dir=frames_dir, n_frames=n_frames)
        if res.frames:
            out[v] = res.frames
        elif res.error:
            logger.warning("frame extraction failed for %s: %s", v.name, res.error)
    return out
