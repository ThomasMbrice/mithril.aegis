"""
Unified Telemetry Plane (UTP) — low-overhead asyncio event bus.

Design goals (from §4 Layer A1):
- Must not become a scaling bottleneck itself.
- All three primitive layers (B, C, D) emit here; the Failure Classifier consumes here.
- Non-blocking publish: queue-full drops event and logs a warning rather than
  back-pressuring the transport fast-path.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Union

from .events import TelemetryEvent

logger = logging.getLogger(__name__)

Subscriber = Callable[[TelemetryEvent], Union[None, Awaitable[None]]]


class UnifiedTelemetryPlane:
    """
    Asyncio-based single event bus.

    Publishers call ``publish()`` (non-blocking, fire-and-forget).
    The background dispatch loop fans each event out to all subscribers in
    registration order, awaiting async subscribers.
    """

    def __init__(self, queue_size: int = 10_000) -> None:
        self._subscribers: list[Subscriber] = []
        self._queue: asyncio.Queue[TelemetryEvent] = asyncio.Queue(maxsize=queue_size)
        self._event_count: int = 0
        self._drop_count: int = 0
        self._running: bool = False
        self._dispatch_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Registration

    def subscribe(self, callback: Subscriber) -> None:
        """Register a subscriber called for every dispatched event."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Subscriber) -> None:
        self._subscribers.remove(callback)

    # ------------------------------------------------------------------
    # Publishing

    async def publish(self, event: TelemetryEvent) -> None:
        """
        Enqueue an event for dispatch.

        Non-blocking.  If the queue is full, the event is dropped and a
        warning is logged — we must never back-pressure the B0 fast-path.
        """
        try:
            self._queue.put_nowait(event)
            self._event_count += 1
        except asyncio.QueueFull:
            self._drop_count += 1
            logger.warning(
                "UTP queue full — dropping %s from rank %d (total drops: %d)",
                event.fault_signal.value,
                event.rank,
                self._drop_count,
            )

    # ------------------------------------------------------------------
    # Lifecycle

    async def start(self) -> None:
        """Start the background dispatch loop."""
        self._running = True
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="utp-dispatch"
        )

    async def stop(self) -> None:
        """
        Gracefully stop: drain pending events, then cancel the task.

        Callers should await this before tearing down the event loop so
        pytest-asyncio doesn't report pending tasks.
        """
        self._running = False
        if self._dispatch_task is not None:
            # Let the loop drain whatever is already queued
            try:
                await asyncio.wait_for(self._drain(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("UTP drain timed out — %d events remain", self._queue.qsize())

            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
            self._dispatch_task = None

    async def _drain(self) -> None:
        """Process the queue until empty."""
        while not self._queue.empty():
            await self._dispatch_one()

    # ------------------------------------------------------------------
    # Internals

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            await self._dispatch_one(event)
            self._queue.task_done()

    async def _dispatch_one(self, event: TelemetryEvent | None = None) -> None:
        if event is None:
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
        for sub in list(self._subscribers):
            try:
                result = sub(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Subscriber %r raised on event %s", sub, event.fault_signal.value
                )

    # ------------------------------------------------------------------
    # Observability

    @property
    def event_count(self) -> int:
        """Total events successfully enqueued since start."""
        return self._event_count

    @property
    def drop_count(self) -> int:
        """Total events dropped due to full queue."""
        return self._drop_count

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
