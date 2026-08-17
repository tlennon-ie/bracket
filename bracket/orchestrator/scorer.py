"""Score a completed run.

Two components, both lower-is-better:
  - loss component:  final_smoothed_loss + |slope_of_last_25%|  (cheap, from tfevents)
  - sample component: 1 - (mean VLM overall / 10)               (slow, from sample images)

Combined score = loss_weight * loss + sample_weight * sample. The judge is
optional; if absent, we fall back to loss-only — same behaviour as v0.2.

Disqualifications return score=+inf and `disqualified` set:
  - tfevents missing or empty
  - NaN final loss
  - positive slope > kill threshold (loss diverging)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

logger = logging.getLogger("bracket.scorer")

from bracket.ema import EMASmoother
from bracket.frame import LossFrame
from bracket.judge.base import JudgeReport, SampleJudge
from bracket.log_loss_reader import parse_log_loss
from bracket.sqlite_loss_reader import parse_sqlite_loss
from bracket.tfevents_reader import TFEventsTail
from bracket.video import (
    VIDEO_EXTENSIONS, extract_all_videos, is_frame_extractable, is_video_file,
    list_video_samples,
)


# sd-scripts writes samples like  candidate_<step>_<idx>_<seed>.png alongside
# a candidate_<step>_<idx>_<seed>.txt that contains the prompt. We pair them
# by stem so the judge knows which prompt produced which image.
_SAMPLE_STEM_RE = re.compile(r"^(?P<base>.+)\.(?P<ext>png|jpg|jpeg|webp)$", re.IGNORECASE)
_VIDEO_STEM_RE = re.compile(r"^(?P<base>.+)\.(?P<ext>mp4|webm|mov|mkv)$", re.IGNORECASE)

# LTX-2 (ltx-trainer) writes validation samples as
# ``step_{step:06d}_{idx+1}.{mp4|png|wav}`` with NO sidecar .txt, where
# ``idx+1`` is a 1-BASED index into the prompts list. When the video pipeline
# extracts frames, the frame stem becomes ``step_{step:06d}_{idx+1}_frameNN``
# (see ``video.extract_frames``), so the trailing ``_frameNN`` is optional.
_LTX2_STEP_STEM_RE = re.compile(
    r"^step_\d+_(?P<one_based_idx>\d+)(?:_frame\d+)?$", re.IGNORECASE
)

# ai-toolkit (ostris) writes validation samples to ``<save_root>/samples/`` as
# ``[time]_{step_num}_[count].{ext}`` where the template substitutes to
# ``<gen_time_ms>__<step_zfill9>_<count>.<jpg|png|webp|...>`` (the ``__`` double
# underscore comes from the empty ``[time]_<step_num>`` join when ``step_num``
# already carries a leading ``_``). ``<count>`` is the 0-BASED prompt index
# (``gen_config.save_image(img, i)`` with ``i`` enumerating the prompts list),
# zero-padded only to ``len(str(max_count))`` which is 1 for the default
# ``max_count=0`` — so a bare ``_0``/``_2``. There is NO sidecar .txt. The
# 13+-digit millisecond timestamp followed by the ``__`` is the distinguishing
# feature that keeps this from colliding with the sd-scripts convention. When
# the video pipeline extracts frames, the stem gains a trailing ``_frameNN``.
_AITK_STEP_STEM_RE = re.compile(
    r"^\d{12,}__\d+_(?P<zero_based_idx>\d+)(?:_frame\d+)?$"
)


@dataclass
class ScoreReport:
    score: float
    components: dict[str, float] = field(default_factory=dict)
    n_steps: int = 0
    disqualified: Optional[str] = None
    judge_report: Optional[JudgeReport] = None


class Scorer:
    def __init__(
        self,
        *,
        ema_alpha: float = 0.05,
        slope_window_fraction: float = 0.25,
        positive_slope_kill_threshold: float = 0.05,
        sample_judge: Optional[SampleJudge] = None,
        loss_weight: float = 0.3,
        sample_weight: float = 0.7,
        clip_iqa_judge: Optional[SampleJudge] = None,
        clip_iqa_dq_threshold: float = 0.30,
    ) -> None:
        if loss_weight < 0 or sample_weight < 0:
            raise ValueError("weights must be >= 0")
        if loss_weight == 0 and sample_weight == 0:
            raise ValueError("at least one of loss_weight / sample_weight must be > 0")
        self.ema_alpha = ema_alpha
        self.slope_window_fraction = slope_window_fraction
        self.positive_slope_kill_threshold = positive_slope_kill_threshold
        self.sample_judge = sample_judge
        self.loss_weight = loss_weight
        self.sample_weight = sample_weight
        # Optional secondary judge used as a DQ gate, NOT as a primary
        # ranker. When set, the scorer judges every image in the sample
        # dir via clip_iqa_judge and disqualifies the run if the median
        # of the per-image scores falls below clip_iqa_dq_threshold.
        # Threshold is in [0, 1] (clip-iqa's native scale).
        self.clip_iqa_judge = clip_iqa_judge
        self.clip_iqa_dq_threshold = clip_iqa_dq_threshold

    def score_tfevents(self, tfevents_path: Optional[Path]) -> ScoreReport:
        """Backwards-compatible loss-only scoring entry point used by tests."""
        if tfevents_path is None or not Path(tfevents_path).exists():
            return ScoreReport(
                score=math.inf, components={}, n_steps=0,
                disqualified="no_tfevents",
            )
        frames: list[LossFrame] = []
        tail = TFEventsTail(
            event_path=str(tfevents_path),
            on_frame=frames.append,
            ema_alpha=self.ema_alpha,
        )
        n = tail.drain_once()
        if n == 0:
            return ScoreReport(
                score=math.inf, components={}, n_steps=0,
                disqualified="empty_tfevents",
            )
        return self._score_frames(frames)

    def score_log_loss(self, loss_log_path: Optional[Path]) -> ScoreReport:
        """Loss-only scoring from a stdout-log file (LTX-2 / ltx-trainer).

        The stdout-log analog of ``score_tfevents``: parse the run's log file
        into LossFrames and run the SAME ``_score_frames`` divergence / slope /
        NaN logic. Disqualifies with ``no_loss_log`` / ``empty_loss_log`` —
        the stdout-log counterparts of ``no_tfevents`` / ``empty_tfevents``.
        """
        if loss_log_path is None or not Path(loss_log_path).exists():
            return ScoreReport(
                score=math.inf, components={}, n_steps=0,
                disqualified="no_loss_log",
            )
        frames = parse_log_loss(loss_log_path, ema_alpha=self.ema_alpha)
        if not frames:
            return ScoreReport(
                score=math.inf, components={}, n_steps=0,
                disqualified="empty_loss_log",
            )
        return self._score_frames(frames)

    def score_sqlite_loss(self, loss_db_path: Optional[Path]) -> ScoreReport:
        """Loss-only scoring from a SQLite ``loss_log.db`` (ai-toolkit / ostris).

        The SQLite analog of ``score_tfevents`` / ``score_log_loss``: parse the
        run's ``loss_log.db`` into LossFrames and run the SAME ``_score_frames``
        divergence / slope / NaN logic. Disqualifies with ``no_loss_db`` /
        ``empty_loss_db`` — the SQLite counterparts of ``no_tfevents`` /
        ``empty_tfevents``. A missing/unopenable DB or one with no loss rows is
        handled by ``parse_sqlite_loss`` returning ``[]`` (it never raises).
        """
        if loss_db_path is None or not Path(loss_db_path).exists():
            return ScoreReport(
                score=math.inf, components={}, n_steps=0,
                disqualified="no_loss_db",
            )
        frames = parse_sqlite_loss(loss_db_path, ema_alpha=self.ema_alpha)
        if not frames:
            return ScoreReport(
                score=math.inf, components={}, n_steps=0,
                disqualified="empty_loss_db",
            )
        return self._score_frames(frames)

    def score_run(
        self,
        *,
        tfevents_path: Optional[Path],
        sample_dir: Optional[Path] = None,
        prompts: Optional[list[str]] = None,
        n_in_dist_prompts: Optional[int] = None,
        loss_log_path: Optional[Path] = None,
        loss_db_path: Optional[Path] = None,
        loss_source: str = "tfevents",
    ) -> ScoreReport:
        """Score a full run: loss component + sample component (if judge + samples).

        The loss component comes from tfevents when ``tfevents_path`` is present
        (every existing trainer). Otherwise the loss curve is parsed from the
        trainer's non-tfevents telemetry and fed through the SAME
        ``_score_frames`` path so the divergence / slope / NaN disqualification
        logic is identical. tfevents is always preferred when both are supplied.
        ``loss_source`` (the candidate's ``LaunchSpec.loss_source``) gates the
        fallback:

          - ``"stdout_log"`` → parse ``loss_log_path`` via ``score_log_loss``
            (LTX-2 / ltx-trainer).
          - ``"sqlite_db"``  → parse ``loss_db_path`` via ``score_sqlite_loss``
            (ai-toolkit / ostris ``loss_log.db``).

        A tfevents trainer that produced no events (crashed/killed before the
        first write) keeps the historical ``no_tfevents`` disqualification — we
        never relabel it ``empty_loss_log`` / ``empty_loss_db`` off another
        source.

        When ``n_in_dist_prompts`` is set, prompts at indices ``[0, n)`` are
        treated as in-distribution and the rest (typically those concatenated
        from ``sample_prompts_ood``) are treated as out-of-distribution. The
        scorer then emits ``in_dist_score`` and ``ood_score`` components so
        downstream consumers (Pareto front, generalisation-gap reports) can
        reason about both axes. The aggregate ``score`` field is unchanged
        for backward compat — it stays the mean across ALL judgements.
        """
        if tfevents_path is not None:
            report = self.score_tfevents(tfevents_path)
        elif loss_source == "stdout_log":
            # ``score_log_loss`` handles a None/missing path itself (→
            # ``no_loss_log``), so the discriminator is purely ``loss_source``.
            report = self.score_log_loss(loss_log_path)
        elif loss_source == "sqlite_db":
            # ``score_sqlite_loss`` handles a None/missing db itself (→
            # ``no_loss_db``), so the discriminator is purely ``loss_source``.
            report = self.score_sqlite_loss(loss_db_path)
        else:
            # tfevents trainer that wrote no events (crashed/killed before the
            # first write). ``tfevents_path`` is None here, so this yields the
            # historical ``no_tfevents`` DQ — NOT ``empty_loss_log`` off the
            # always-present run log. Keeps ledger/report reasons consistent.
            report = self.score_tfevents(tfevents_path)
        if report.disqualified is not None:
            return report
        if self.sample_judge is None or sample_dir is None or not prompts:
            # Loss-only — already in `report`. Tag the components.
            report.components["loss_only"] = 1.0
            return report

        # If the trainer emitted videos (Wan / Hunyuan / LTX / FramePack),
        # extract a small set of representative frames before judging. The
        # extracted PNGs live in `sample_dir/_frames/` so the same pairing
        # logic (sidecar .txt or filename idx) finds them.
        videos = list_video_samples(sample_dir)
        if videos:
            frames_dir = sample_dir / "_frames"
            extracted = extract_all_videos(videos, frames_dir=frames_dir)
            for video, frames in extracted.items():
                sidecar = video.with_suffix(".txt")
                if not sidecar.exists():
                    continue
                # Replicate the prompt sidecar next to each extracted frame so
                # _pair_samples_with_prompts picks it up via the .txt branch.
                for frame in frames:
                    frame.with_suffix(".txt").write_text(
                        sidecar.read_text(encoding="utf-8", errors="replace"),
                        encoding="utf-8",
                    )

        prompt_for_image = _pair_samples_with_prompts(sample_dir, prompts)
        if not prompt_for_image:
            # Trainer wrote no samples — fall back to loss-only. Surface the
            # condition: silent fallback historically hid pruned-before-first-
            # sample runs and sample_every_n_steps misconfigurations.
            logger.warning(
                "scorer: no sample images found in %s — falling back to "
                "loss-only score. Likely causes: run killed before the first "
                "sample step, sample_every_n_steps too high for max_steps, "
                "or trainer wrote samples to an unexpected location.",
                sample_dir,
            )
            report.components["loss_only"] = 1.0
            report.components["sample_dir_empty"] = 1.0
            return report

        judge_report = self.sample_judge.judge_run(sample_dir, prompt_for_image)
        loss_score = report.score
        sample_score = judge_report.normalized_score   # already lower-is-better
        combined = self.loss_weight * loss_score + self.sample_weight * sample_score
        denom = self.loss_weight + self.sample_weight
        if denom > 0:
            combined /= denom

        new_components = dict(report.components)
        new_components.update({
            "loss_score": loss_score,
            "sample_score": sample_score,
            "sample_overall_0_10": judge_report.mean_overall,
            "sample_prompt_adherence_0_10": judge_report.mean_prompt_adherence,
            "sample_visual_quality_0_10": judge_report.mean_visual_quality,
            "sample_artifact_free_0_10": judge_report.mean_artifact_free,
            "samples_judged": float(judge_report.n_images),
            "samples_failed": float(judge_report.n_failed),
        })
        # Split in-dist vs OOD scores when the caller flagged a split. The
        # split is by prompt string membership — the trainer's filename
        # convention only encodes a prompt index, but by this point each
        # judgement carries the actual prompt text it was paired with.
        if (
            n_in_dist_prompts is not None
            and n_in_dist_prompts >= 0
            and prompts
            and n_in_dist_prompts < len(prompts)
        ):
            in_dist_prompts = set(prompts[:n_in_dist_prompts])
            in_dist_scores = [
                j.overall for j in judge_report.judgements
                if j.error is None and j.prompt in in_dist_prompts
            ]
            ood_scores = [
                j.overall for j in judge_report.judgements
                if j.error is None and j.prompt not in in_dist_prompts
            ]
            # Mirror the aggregate's lower-is-better mapping so callers can
            # compose these scores with the loss component without a sign flip.
            if in_dist_scores:
                in_mean = sum(in_dist_scores) / len(in_dist_scores)
                new_components["in_dist_score"] = 1.0 - (in_mean / 10.0)
                new_components["in_dist_overall_0_10"] = in_mean
                new_components["in_dist_samples_judged"] = float(len(in_dist_scores))
            if ood_scores:
                ood_mean = sum(ood_scores) / len(ood_scores)
                new_components["ood_score"] = 1.0 - (ood_mean / 10.0)
                new_components["ood_overall_0_10"] = ood_mean
                new_components["ood_samples_judged"] = float(len(ood_scores))
            if in_dist_scores and ood_scores:
                # Positive value = OOD performance worse than in-dist
                # (lower-is-better space), i.e. the model failed to generalise.
                new_components["generalization_gap"] = (
                    new_components["ood_score"] - new_components["in_dist_score"]
                )
        # CLIP-IQA gate. Runs alongside the VLM judge as an additional
        # check; emits a clip_iqa_median component and DQs if the
        # median falls below the configured threshold. The VLM's score
        # is the primary ranker — clip-iqa just catches "melted"
        # outputs the VLM happily called fine.
        if self.clip_iqa_judge is not None and prompt_for_image:
            disqualified, clip_components = self._run_clip_iqa_gate(prompt_for_image)
            new_components.update(clip_components)
            if disqualified is not None:
                return ScoreReport(
                    score=math.inf, components=new_components,
                    n_steps=report.n_steps, disqualified=disqualified,
                    judge_report=judge_report,
                )
        return ScoreReport(
            score=combined, components=new_components,
            n_steps=report.n_steps, disqualified=None, judge_report=judge_report,
        )

    def _run_clip_iqa_gate(
        self, prompt_for_image: dict[Path, str],
    ) -> tuple[Optional[str], dict[str, float]]:
        """Run the configured CLIP-IQA judge across the run's samples
        and return (``disqualified`` reason, components dict)."""
        if self.clip_iqa_judge is None:
            return None, {}
        scores_01: list[float] = []
        for img, prompt in prompt_for_image.items():
            j = self.clip_iqa_judge.judge_image(img, prompt)
            if j.error is not None:
                continue
            # ClipIqaJudge encodes the raw [0,1] value as overall * 10.
            scores_01.append(max(0.0, min(1.0, j.overall / 10.0)))
        # Free the model's VRAM ahead of the next training run.
        try:
            self.clip_iqa_judge.eject()
        except Exception:  # noqa: BLE001 — eject must never block the loop
            pass
        if not scores_01:
            return None, {"clip_iqa_unavailable": 1.0}
        median = _percentile(scores_01, 50.0)
        components = {
            "clip_iqa_median": float(median),
            "clip_iqa_samples_judged": float(len(scores_01)),
        }
        if median < float(self.clip_iqa_dq_threshold):
            return "low_clip_iqa", components
        return None, components

    def _score_frames(self, frames: list[LossFrame]) -> ScoreReport:
        n = len(frames)
        last = frames[-1]
        if math.isnan(last.smoothed_loss) or math.isnan(last.raw_loss):
            return ScoreReport(
                score=math.inf,
                components={"final_smoothed": last.smoothed_loss, "final_raw": last.raw_loss},
                n_steps=n,
                disqualified="nan_loss",
            )
        window = max(2, int(round(self.slope_window_fraction * n)))
        tail = frames[-window:]
        slope = _least_squares_slope([f.step for f in tail], [f.smoothed_loss for f in tail])
        components: dict[str, float] = {
            "final_smoothed": last.smoothed_loss,
            "final_raw": last.raw_loss,
            "slope": slope,
            "slope_window": float(window),
            "first_smoothed": frames[0].smoothed_loss,
            "n_frames": float(n),
        }

        # Gradient-norm stability — only emitted by trainers that route a
        # ``mean_grad_norm`` value through ``generate_step_logs``. When
        # nothing is logged we skip the components entirely; absence of a
        # signal is not a divergence signal.
        grad_values = [f.grad_norm for f in frames if f.grad_norm is not None]
        if grad_values:
            # Explosion check: any non-finite value or a 99th-percentile that
            # blew past the hard cap means optimisation has gone unstable —
            # the rest of the loss curve is unreliable noise.
            if any(not math.isfinite(g) for g in grad_values):
                gn_p99 = math.inf
            else:
                gn_p99 = _percentile(grad_values, 99.0)
            components["grad_norm_p99"] = gn_p99

            window_size = max(2, int(round(self.slope_window_fraction * n)))
            tail_grads = [
                f.grad_norm for f in frames[-window_size:] if f.grad_norm is not None
            ]
            if len(tail_grads) >= 2:
                mean_tail = sum(tail_grads) / len(tail_grads)
                variance = sum((g - mean_tail) ** 2 for g in tail_grads) / len(tail_grads)
                std_tail = math.sqrt(variance)
                if mean_tail > 0 and math.isfinite(mean_tail) and math.isfinite(std_tail):
                    components["grad_norm_oscillation"] = std_tail / mean_tail
                else:
                    components["grad_norm_oscillation"] = math.inf if not math.isfinite(std_tail) else 0.0

            if not math.isfinite(gn_p99) or gn_p99 > 100.0:
                return ScoreReport(
                    score=math.inf, components=components, n_steps=n,
                    disqualified="gradient_explosion",
                )

        if slope > self.positive_slope_kill_threshold:
            return ScoreReport(
                score=math.inf, components=components, n_steps=n,
                disqualified=f"diverging(slope={slope:.4f})",
            )
        score = last.smoothed_loss + abs(slope)
        return ScoreReport(score=score, components=components, n_steps=n)


def _pair_samples_with_prompts(
    sample_dir: Path, prompts: list[str]
) -> dict[Path, str]:
    """Map each sampled image file → the prompt that produced it.

    Two trainer filename conventions are supported:

      sd-scripts:  <output_name>_<step>_<idx>_<seed>.png
      musubi:      <output_name>_<step>_<idx>_<YYYYMMDDhhmmss>_<seed>_<batch>.png
                   (step may be prefixed `e` for end-of-epoch samples)

    In both cases <idx> is the 0-based index into the sample_prompts file
    (skipping comments). When a sidecar .txt is present we use that instead.
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        return {}
    out: dict[Path, str] = {}
    # Walk one level deep so the `_frames/` subdir produced by video samplers
    # (extract_all_videos) is included.
    candidates: list[Path] = []
    for entry in sorted(sample_dir.iterdir()):
        if entry.is_file():
            candidates.append(entry)
        elif entry.is_dir() and entry.name == "_frames":
            candidates.extend(sorted(p for p in entry.iterdir() if p.is_file()))
    for img in candidates:
        m = _SAMPLE_STEM_RE.match(img.name)
        if not m:
            continue
        # An animated WebP/GIF matches the still-image regex but has already
        # been frame-extracted into `_frames/`. Judging the container too would
        # score its first frame a second time and skew the run's mean.
        if is_frame_extractable(img):
            continue
        sidecar = img.with_suffix(".txt")
        if sidecar.exists():
            try:
                out[img] = sidecar.read_text(encoding="utf-8", errors="replace").strip()
                continue
            except OSError:
                pass
        idx = _extract_prompt_idx(m.group("base"))
        if idx is None:
            continue
        if 0 <= idx < len(prompts):
            out[img] = prompts[idx]
    return out


def _extract_prompt_idx(stem: str) -> Optional[int]:
    """Return the 0-based prompt index from a sample filename stem, or None."""
    # LTX-2 (ltx-trainer): ``step_<step>_<idx+1>`` with a 1-BASED index, and an
    # optional ``_frameNN`` suffix when the index survives video-frame
    # extraction. Checked first because the bare-int branches below would read
    # the 1-based index as if it were 0-based.
    ltx = _LTX2_STEP_STEM_RE.match(stem)
    if ltx is not None:
        one_based = int(ltx.group("one_based_idx"))
        if one_based >= 1:
            return one_based - 1
    # ai-toolkit (ostris): ``<gen_time_ms>__<step_zfill9>_<count>`` with a
    # 0-BASED ``count`` (optional ``_frameNN`` for extracted video frames).
    # Checked before the sd-scripts ``parts[-2]`` branch, which would otherwise
    # mistake the zero-padded step for the index.
    aitk = _AITK_STEP_STEM_RE.match(stem)
    if aitk is not None:
        return int(aitk.group("zero_based_idx"))
    parts = stem.split("_")
    # musubi: idx sits one position before a 14-digit timestamp segment.
    for i, p in enumerate(parts):
        if i > 0 and len(p) == 14 and p.isdigit() and parts[i - 1].isdigit():
            return int(parts[i - 1])
    # sd-scripts: <name>_<step>_<idx>_<seed> → idx is parts[-2].
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return int(parts[-2])
    return None


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches numpy default). ``values``
    is sorted internally; caller does not need to pre-sort."""
    if not values:
        return float("nan")
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    k = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    frac = k - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _least_squares_slope(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den
