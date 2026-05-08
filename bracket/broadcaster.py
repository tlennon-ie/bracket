"""Fan one LossFrame out to N WebSocket subscribers without blocking producers.

Each subscriber gets a bounded asyncio.Queue. If a subscriber is slow, frames
to *that subscriber* are dropped, never blocking the polling thread or other
subscribers. Producers (the tfevents tail thread) submit frames via the
thread-safe submit() method, which schedules a put on the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from bracket.frame import LossFrame

logger = logging.getLogger(__name__)


class Broadcaster:
    def __init__(self, queue_max: int = 256) -> None:
        self._queue_max = queue_max
        self._subscribers: set[asyncio.Queue[LossFrame]] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the broadcaster to the running event loop. Call once during
        startup, before any background producer thread starts emitting frames."""
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue[LossFrame]:
        q: asyncio.Queue[LossFrame] = asyncio.Queue(maxsize=self._queue_max)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[LossFrame]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    def submit(self, frame: LossFrame) -> None:
        """Thread-safe: schedule the fan-out on the bound event loop. Drops the
        frame if no loop is bound (e.g. during shutdown)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._fanout_nowait, frame)
        except RuntimeError:
            # event loop closed mid-call; safe to ignore — same as no-op above
            pass

    def _fanout_nowait(self, frame: LossFrame) -> None:
        # Iterate a snapshot so concurrent unsubscribe doesn't break iteration
        for q in list(self._subscribers):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                # Drop the *oldest* frame instead of the newest — a slow
                # subscriber should still see recent state, just at lower fps.
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
