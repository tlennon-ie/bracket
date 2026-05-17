"""Tail a TensorBoard event file and emit LossFrames as new steps land.

The musubi_tuner training loop calls accelerator.log({...}, step=global_step)
which routes through accelerate's TensorBoardTracker, which writes scalar
events into events.out.tfevents.* via tf.summary.

We poll the file with EventAccumulator.Reload() — it's cheap (incremental on
the underlying RecordReader) and the API is stable. For v0.1 this is the
right tradeoff: zero-touch on the trainer, works on completed runs *and* live
runs, no need to install a custom callback.

v0.1 limitations (documented; v0.2 fixes):
  * Single file only — does not pick up rotation if accelerate opens a new
    tfevents file mid-run. In practice musubi_tuner writes one file per run.
  * No timestep info — loss is already batch-meaned by the time it's logged.
  * Polling cadence is 1s by default; for sub-second visibility, use the v0.2
    in-process hook.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from bracket.ema import EMASmoother
from bracket.frame import LossFrame

logger = logging.getLogger(__name__)

# Per-step scalar tag the trainer emits. Different trainers use different
# tag names:
#   * musubi-tuner (Z-Image / Flux-2-Klein / Wan / Hunyuan / Qwen-Image)
#     writes ``loss/current``.
#   * sd-scripts SDXL/SD3.5/Lumina/Anima writes a bare ``loss`` plus
#     ``loss/epoch``. It does *not* emit ``loss/average``.
#   * sd-scripts Flux.1 LoRA writes ``loss/average`` (kept for compat).
# We try each candidate and use the first one present in the file.
# ``LOSS_TAG`` stays as a single-string public constant for backwards
# compatibility with old callers.
LOSS_TAG = "loss/current"
LOSS_TAG_CANDIDATES: tuple[str, ...] = ("loss/current", "loss/average", "loss")
LR_TAG = "lr/unet"
LR_TAG_CANDIDATES: tuple[str, ...] = ("lr/unet", "lr/textencoder")
# Per-step gradient norm. Different trainers use different names:
#   * sd-scripts/train_network.py emits ``norm/avg_grad_norm`` when the
#     ``mean_grad_norm`` arg is passed to ``generate_step_logs`` (LoRA flow).
#   * musubi-tuner mirrors the sd-scripts log dict, so the same tag shows up
#     for Z-Image / Flux-2-Klein / Qwen-Image / Hunyuan when grad-norm is
#     plumbed through.
#   * Some forks log directly under ``train/grad_norm`` or a bare
#     ``grad_norm``. Keep both as fallbacks.
# When no candidate tag is present (e.g. SDXL full-FT today) the reader
# simply emits ``LossFrame.grad_norm=None`` and the scorer skips its
# stability components.
GRAD_NORM_TAG = "norm/avg_grad_norm"
GRAD_NORM_TAG_CANDIDATES: tuple[str, ...] = (
    "norm/avg_grad_norm",
    "train/grad_norm",
    "grad_norm",
    "loss/grad_norm",
)


def scalar_tags_in(event_path: str) -> list[str]:
    """Return the list of scalar tags present in an event file. Useful for
    smoke-testing that we're pointed at the right file before starting a tail."""
    if not os.path.exists(event_path):
        raise FileNotFoundError(event_path)
    ea = EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    return list(ea.Tags()["scalars"])


class TFEventsTail:
    """Polls an event file in a background thread and emits LossFrames.

    The on_frame callback is invoked from the polling thread. Keep it cheap —
    typical pattern is to push to an asyncio queue and let the broadcaster fan
    out from the event loop.
    """

    def __init__(
        self,
        event_path: str,
        on_frame: Callable[[LossFrame], None],
        *,
        poll_interval_s: float = 1.0,
        ema_alpha: float = 0.05,
        loss_tag: str | tuple[str, ...] = LOSS_TAG_CANDIDATES,
        lr_tag: str | tuple[str, ...] = LR_TAG_CANDIDATES,
        grad_norm_tag: str | tuple[str, ...] = GRAD_NORM_TAG_CANDIDATES,
    ) -> None:
        if not os.path.exists(event_path):
            raise FileNotFoundError(event_path)
        self._event_path = event_path
        self._on_frame = on_frame
        self._poll_interval_s = poll_interval_s
        self._loss_tag_candidates = (loss_tag,) if isinstance(loss_tag, str) else loss_tag
        self._lr_tag_candidates = (lr_tag,) if isinstance(lr_tag, str) else lr_tag
        self._grad_norm_tag_candidates = (
            (grad_norm_tag,) if isinstance(grad_norm_tag, str) else grad_norm_tag
        )
        self._smoother = EMASmoother(alpha=ema_alpha)
        self._last_step_emitted = -1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("already started")
        self._thread = threading.Thread(
            target=self._run, name="tfevents-tail", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        ea = EventAccumulator(self._event_path, size_guidance={"scalars": 0})
        while not self._stop.is_set():
            try:
                ea.Reload()
                self._drain(ea)
            except Exception:
                logger.exception("tfevents tail iteration failed; continuing")
            self._stop.wait(self._poll_interval_s)

    def drain_once(self) -> int:
        """Read everything currently in the file and emit new frames. Returns
        count of frames emitted. Useful for tests and for processing finished
        runs in one shot."""
        ea = EventAccumulator(self._event_path, size_guidance={"scalars": 0})
        ea.Reload()
        return self._drain(ea)

    def _drain(self, ea: EventAccumulator) -> int:
        tags = ea.Tags()["scalars"]
        loss_tag = next((t for t in self._loss_tag_candidates if t in tags), None)
        if loss_tag is None:
            return 0
        loss_events = ea.Scalars(loss_tag)
        lr_tag = next((t for t in self._lr_tag_candidates if t in tags), None)
        lr_events = {e.step: e.value for e in ea.Scalars(lr_tag)} if lr_tag else {}
        grad_norm_tag = next(
            (t for t in self._grad_norm_tag_candidates if t in tags), None
        )
        grad_norm_events: dict[int, float] = (
            {e.step: e.value for e in ea.Scalars(grad_norm_tag)}
            if grad_norm_tag
            else {}
        )
        emitted = 0
        for ev in loss_events:
            if ev.step <= self._last_step_emitted:
                continue
            smoothed = self._smoother.update(ev.value)
            frame = LossFrame(
                step=ev.step,
                raw_loss=float(ev.value),
                smoothed_loss=float(smoothed),
                lr=float(lr_events.get(ev.step)) if ev.step in lr_events else None,
                wall_time=float(ev.wall_time),
                timestep=None,
                timestep_bucket=None,
                grad_norm=(
                    float(grad_norm_events[ev.step])
                    if ev.step in grad_norm_events
                    else None
                ),
            )
            try:
                self._on_frame(frame)
            except Exception:
                logger.exception("on_frame callback raised; dropping frame %d", ev.step)
            self._last_step_emitted = ev.step
            emitted += 1
        return emitted
