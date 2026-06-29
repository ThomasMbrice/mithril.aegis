"""
UT-A1 — Unified Telemetry Plane.

Pass criteria (§5.1):
  - No event loss under sustained injection
  - Queue-full drops gracefully (no crash, warning logged)
  - Async subscribers are awaited correctly
  - Multiple subscribers all receive every event
"""

from __future__ import annotations

import asyncio

import pytest

from aegis.telemetry.bus import UnifiedTelemetryPlane
from aegis.telemetry.events import FaultSignal, TelemetryEvent


def _event(rank: int = 0, signal: FaultSignal = FaultSignal.NIC_PORT_FLAP) -> TelemetryEvent:
    return TelemetryEvent(
        rank=rank,
        node="node0",
        fault_signal=signal,
        raw_payload={},
        epoch=0,
    )


async def test_basic_publish_and_receive() -> None:
    utp = UnifiedTelemetryPlane()
    received: list[TelemetryEvent] = []
    utp.subscribe(received.append)

    await utp.start()
    await utp.publish(_event())
    await asyncio.sleep(0.1)
    await utp.stop()

    assert len(received) == 1
    assert received[0].fault_signal == FaultSignal.NIC_PORT_FLAP


async def test_multiple_subscribers_all_receive() -> None:
    utp = UnifiedTelemetryPlane()
    buckets: list[list[TelemetryEvent]] = [[], [], []]
    for bucket in buckets:
        utp.subscribe(bucket.append)

    await utp.start()
    for i in range(5):
        await utp.publish(_event(rank=i))
    await asyncio.sleep(0.1)
    await utp.stop()

    for bucket in buckets:
        assert len(bucket) == 5


async def test_no_event_loss_under_burst() -> None:
    """UT-A1: inject burst, verify no loss."""
    N = 500
    utp = UnifiedTelemetryPlane(queue_size=N * 2)
    received: list[TelemetryEvent] = []
    utp.subscribe(received.append)

    await utp.start()
    for i in range(N):
        await utp.publish(_event(rank=i % 8))
    await asyncio.sleep(0.5)
    await utp.stop()

    assert len(received) == N, f"Lost {N - len(received)} events"


async def test_queue_full_drops_gracefully() -> None:
    """Queue overflow must not raise — just drop and count."""
    utp = UnifiedTelemetryPlane(queue_size=2)
    received: list[TelemetryEvent] = []

    async def slow_sub(event: TelemetryEvent) -> None:
        await asyncio.sleep(0.05)
        received.append(event)

    utp.subscribe(slow_sub)
    await utp.start()

    # publish more than the queue can hold immediately
    for _ in range(20):
        await utp.publish(_event())  # must not raise

    await asyncio.sleep(0.5)
    await utp.stop()

    # Some events were dropped, but no exception occurred
    assert utp.drop_count > 0
    assert len(received) <= 20


async def test_async_subscriber_is_awaited() -> None:
    utp = UnifiedTelemetryPlane()
    received: list[TelemetryEvent] = []

    async def async_sub(event: TelemetryEvent) -> None:
        await asyncio.sleep(0)
        received.append(event)

    utp.subscribe(async_sub)
    await utp.start()
    await utp.publish(_event())
    await asyncio.sleep(0.1)
    await utp.stop()

    assert len(received) == 1


async def test_event_count_tracks_published() -> None:
    utp = UnifiedTelemetryPlane()
    await utp.start()
    for _ in range(10):
        await utp.publish(_event())
    await asyncio.sleep(0.1)
    await utp.stop()

    assert utp.event_count == 10


async def test_subscriber_exception_does_not_stop_dispatch() -> None:
    """A crashing subscriber must not prevent other subscribers from receiving."""
    utp = UnifiedTelemetryPlane()
    good_received: list[TelemetryEvent] = []

    def bad_sub(event: TelemetryEvent) -> None:
        raise RuntimeError("subscriber kaboom")

    utp.subscribe(bad_sub)
    utp.subscribe(good_received.append)

    await utp.start()
    await utp.publish(_event())
    await asyncio.sleep(0.1)
    await utp.stop()

    assert len(good_received) == 1
