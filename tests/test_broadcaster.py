import asyncio

import pytest

from bracket.broadcaster import Broadcaster
from bracket.frame import LossFrame


def _frame(step: int) -> LossFrame:
    return LossFrame(
        step=step, raw_loss=0.3, smoothed_loss=0.3, lr=1e-6, wall_time=0.0,
        timestep=None, timestep_bucket=None, grad_norm=None,
    )


async def test_subscribe_and_receive():
    b = Broadcaster()
    b.bind_loop(asyncio.get_running_loop())
    q = await b.subscribe()
    b.submit(_frame(1))
    b.submit(_frame(2))
    await asyncio.sleep(0.05)
    f1 = await asyncio.wait_for(q.get(), timeout=0.5)
    f2 = await asyncio.wait_for(q.get(), timeout=0.5)
    assert (f1.step, f2.step) == (1, 2)


async def test_two_subscribers_both_receive():
    b = Broadcaster()
    b.bind_loop(asyncio.get_running_loop())
    a = await b.subscribe()
    c = await b.subscribe()
    b.submit(_frame(7))
    await asyncio.sleep(0.05)
    fa = await asyncio.wait_for(a.get(), timeout=0.5)
    fc = await asyncio.wait_for(c.get(), timeout=0.5)
    assert fa.step == 7 and fc.step == 7


async def test_full_queue_drops_oldest_not_newest():
    b = Broadcaster(queue_max=2)
    b.bind_loop(asyncio.get_running_loop())
    q = await b.subscribe()
    for s in range(1, 6):
        b.submit(_frame(s))
    await asyncio.sleep(0.05)
    # Queue capacity 2 → only the two most recent frames survive
    received = []
    while not q.empty():
        received.append(q.get_nowait().step)
    assert received == [4, 5]


async def test_submit_with_no_loop_bound_is_safe():
    b = Broadcaster()
    # Never call bind_loop — submit should silently no-op rather than crash
    b.submit(_frame(1))


async def test_unsubscribe_stops_delivery():
    b = Broadcaster()
    b.bind_loop(asyncio.get_running_loop())
    q = await b.subscribe()
    await b.unsubscribe(q)
    b.submit(_frame(1))
    await asyncio.sleep(0.05)
    assert q.empty()
    assert b.subscriber_count == 0
