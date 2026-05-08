"""FastAPI app exposing /ws/loss and /health, fed by a TFEventsTail.

The app object is built by build_app(event_path, ...) so callers (CLI, tests,
embedding hosts) can wire in their own paths and EMA settings.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from bracket.broadcaster import Broadcaster
from bracket.tfevents_reader import TFEventsTail

logger = logging.getLogger(__name__)


def build_app(
    event_path: str,
    *,
    ema_alpha: float = 0.05,
    poll_interval_s: float = 1.0,
    queue_max: int = 256,
) -> FastAPI:
    broadcaster = Broadcaster(queue_max=queue_max)
    tail: Optional[TFEventsTail] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal tail
        broadcaster.bind_loop(asyncio.get_running_loop())
        tail = TFEventsTail(
            event_path=event_path,
            on_frame=broadcaster.submit,
            poll_interval_s=poll_interval_s,
            ema_alpha=ema_alpha,
        )
        tail.start()
        logger.info("tfevents tail started on %s", event_path)
        try:
            yield
        finally:
            if tail is not None:
                tail.stop()
                logger.info("tfevents tail stopped")

    app = FastAPI(title="bracket", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "subscribers": broadcaster.subscriber_count,
            "event_path": event_path,
        }

    @app.websocket("/ws/loss")
    async def ws_loss(ws: WebSocket) -> None:
        await ws.accept()
        q = await broadcaster.subscribe()
        try:
            while True:
                frame = await q.get()
                await ws.send_json(frame.to_json())
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("ws_loss handler error")
        finally:
            await broadcaster.unsubscribe(q)

    return app
