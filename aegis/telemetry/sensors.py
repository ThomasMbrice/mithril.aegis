"""
Sensor abstractions — wrap each primitive's native detector as a UTP publisher.

Design decision (§3.1): each primitive's native detector becomes a *sensor*
that publishes to UTP rather than acting unilaterally.  The FC, not the
primitive, decides the response.
"""

from __future__ import annotations

import abc

from .bus import UnifiedTelemetryPlane
from .events import TelemetryEvent


class SensorBase(abc.ABC):
    """
    Abstract fault sensor.

    Subclasses wrap R²CCL's NIC detector, MeCeFO's node-death hook,
    or TierCheck's hung-rank probe and convert their native signals to
    ``TelemetryEvent`` objects on the UTP.
    """

    def __init__(self, utp: UnifiedTelemetryPlane, source: str) -> None:
        self._utp = utp
        self._source = source

    async def emit(self, event: TelemetryEvent) -> None:
        """Publish a fault event to the UTP."""
        await self._utp.publish(event)

    @abc.abstractmethod
    async def start(self) -> None:
        """Begin monitoring."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop monitoring."""


class SyntheticSensor(SensorBase):
    """
    Emits pre-defined events — used by ``chaos_inject`` and unit tests.

    In production this is replaced by real sensors wired to NCCL callbacks,
    CUDA device-lost events, and heartbeat timeouts.
    """

    def __init__(self, utp: UnifiedTelemetryPlane) -> None:
        super().__init__(utp, source="synthetic")

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def inject(self, event: TelemetryEvent) -> None:
        """Publish a synthetic event (shorthand for tests)."""
        await self.emit(event)
