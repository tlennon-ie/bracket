"""Parse a stdout-log training loss curve into LossFrames.

Some trainers don't write TensorBoard tfevents at all — they only log loss to
stdout. Lightricks' ``ltx-trainer`` (LTX-2) is the first such backend Bracket
drives: when launched with ``--disable-progress-bars`` its ``LtxvTrainer.train()``
emits one newline-terminated line every 20 optimization steps, e.g.::

    Step 20/2000 - Loss: 0.1234, LR: 1.00e-04, Time/Step: 2.31s, Total Time: 0:01:23

Bracket's runner redirects that stdout straight into a per-run log file (which
also carries leading ``#`` header lines the runner writes, plus arbitrary
trainer chatter). This reader is the stdout-log analog of
``bracket.tfevents_reader``: it scans a completed log file, keeps only the
lines matching the ``Step N/M - Loss: ...`` shape, and produces the same
``LossFrame`` schema (EMA-smoothed, tfevents-only fields left ``None``) so the
scorer can reuse its existing ``_score_frames`` path unchanged.

Unlike ``TFEventsTail`` this reader is post-hoc only — there is no live tail
for stdout-log runs yet (live pruning stays tfevents-based).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from bracket.ema import EMASmoother
from bracket.frame import LossFrame

logger = logging.getLogger("bracket.log_loss_reader")


# Matches the cadence line from ``LtxvTrainer.train()``. We anchor on the
# ``Step <step>/<total>`` prefix and the ``Loss: <float>`` field, both of
# which are stable. The ``LR: <float>`` field is optional (older builds /
# alternative log configs may omit it), so it lives in its own optional group
# and is searched separately from the strict step/loss match. Spacing and the
# trailing time fields are tolerated by not anchoring the end of the line.
#
# The loss group also matches ``nan`` / ``inf`` / ``-inf`` (PyTorch trainers
# emit these verbatim when gradients explode — ``f"{float('nan'):.4f}"`` →
# ``"nan"``). Capturing them is deliberate: ``float("nan")`` / ``float("inf")``
# propagate into the LossFrame so the scorer's ``_score_frames`` NaN/Inf
# disqualification fires (``nan_loss`` / ``gradient_explosion``). If we dropped
# them, a diverged run would be silently scored off its last finite frame.
_STEP_LOSS_RE = re.compile(
    r"\bStep\s+(?P<step>\d+)\s*/\s*\d+\s*-\s*Loss:\s*"
    r"(?P<loss>[-+]?(?:nan|inf(?:inity)?|\d*\.?\d+(?:[eE][-+]?\d+)?))",
    re.IGNORECASE,
)
_LR_RE = re.compile(
    r"\bLR:\s*(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


class LogLossReader:
    """Parse a completed stdout-log file into a list of LossFrames.

    Lines that don't match the ``Step N/M - Loss: ...`` pattern (``#`` headers,
    trainer banners, validation chatter) are skipped silently. Frames are
    de-duplicated by step and returned in ascending step order — if the same
    step appears twice (e.g. a resumed run re-logs it) the first occurrence
    wins, mirroring ``TFEventsTail``'s monotonic ``_last_step_emitted`` guard.
    """

    def __init__(self, log_path: Path | str, *, ema_alpha: float = 0.05) -> None:
        self._log_path = Path(log_path)
        self._ema_alpha = ema_alpha

    def read_all(self) -> list[LossFrame]:
        if not self._log_path.exists():
            logger.debug("log loss file does not exist: %s", self._log_path)
            return []
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.exception("failed to read log loss file %s", self._log_path)
            return []

        # First pass: collect (step -> (raw_loss, lr)), first occurrence wins.
        parsed: dict[int, tuple[float, Optional[float]]] = {}
        for line in text.splitlines():
            m = _STEP_LOSS_RE.search(line)
            if m is None:
                continue
            try:
                step = int(m.group("step"))
                raw_loss = float(m.group("loss"))
            except (TypeError, ValueError):
                continue
            if step in parsed:
                continue
            lr_match = _LR_RE.search(line)
            lr: Optional[float] = None
            if lr_match is not None:
                try:
                    lr = float(lr_match.group("lr"))
                except (TypeError, ValueError):
                    lr = None
            parsed[step] = (raw_loss, lr)

        # Second pass: build frames in ascending step order, feeding the EMA in
        # step order so the smoothed curve matches the tfevents reader.
        smoother = EMASmoother(alpha=self._ema_alpha)
        wall = time.time()
        frames: list[LossFrame] = []
        for step in sorted(parsed):
            raw_loss, lr = parsed[step]
            smoothed = smoother.update(raw_loss)
            frames.append(
                LossFrame(
                    step=step,
                    raw_loss=float(raw_loss),
                    smoothed_loss=float(smoothed),
                    lr=lr,
                    wall_time=float(wall),
                    timestep=None,
                    timestep_bucket=None,
                    grad_norm=None,
                )
            )
        return frames


def parse_log_loss(
    log_path: Path | str, *, ema_alpha: float = 0.05,
) -> list[LossFrame]:
    """Module-level convenience wrapper around ``LogLossReader.read_all()``."""
    return LogLossReader(log_path, ema_alpha=ema_alpha).read_all()
