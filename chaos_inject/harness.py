"""
ChaosHarness — inject synthetic faults into the AEGIS Unified Telemetry Plane.

Build-order note (§5): this harness is a first-class tool, not a test helper.
It drives:
  - Unit tests (UT-A1–A3, UT-B/C/D)
  - Integration tests (IT-1 through IT-6)
  - Scale / KPI tests (ST-1–ST-3) via inject_rate()

All injection goes through the UTP so the full EPE → layer pipeline is
exercised exactly as in production.
"""

from __future__ import annotations

import asyncio
import logging
import time

from aegis.epoch.service import FaultEpochService
from aegis.telemetry.bus import UnifiedTelemetryPlane
from aegis.telemetry.events import TelemetryEvent

from .faults import BurstSpec, FaultSpec

logger = logging.getLogger(__name__)


class ChaosHarness:
    """
    Fault injection harness.

    Instantiate with a live UTP and epoch service (e.g., from AegisRuntime),
    then call inject() / inject_burst() / inject_rate() from async tests.
    """

    def __init__(
        self,
        utp: UnifiedTelemetryPlane,
        epoch_service: FaultEpochService,
    ) -> None:
        self._utp = utp
        self._epochs = epoch_service
        self._injected: list[TelemetryEvent] = []

    # ------------------------------------------------------------------
    # Injection API

    async def inject(self, spec: FaultSpec) -> list[TelemetryEvent]:
        """
        Inject a fault (or repeated faults) into the UTP.

        Returns the list of events that were published.
        """
        if spec.delay_secs > 0:
            await asyncio.sleep(spec.delay_secs)

        events: list[TelemetryEvent] = []
        for i in range(spec.repeat):
            if i > 0 and spec.repeat_interval_secs > 0:
                await asyncio.sleep(spec.repeat_interval_secs)

            event = TelemetryEvent(
                rank=spec.rank,
                node=spec.node,
                fault_signal=spec.fault_signal,
                raw_payload={"injected": True, "repetition": i},
                epoch=self._epochs.current(),
                nic_id=spec.nic_id,
                rack_id=spec.rack_id,
                source="chaos_inject",
            )
            await self._utp.publish(event)
            self._injected.append(event)
            events.append(event)

            logger.debug(
                "[chaos] %s rank=%d node=%s (%d/%d)",
                spec.fault_signal.value, spec.rank, spec.node, i + 1, spec.repeat,
            )

        return events

    async def inject_burst(self, burst: BurstSpec) -> list[TelemetryEvent]:
        """
        Inject a correlated burst of faults with controlled inter-fault delay.

        Used for IT-3 (B1 burst → B4 re-classification) and IT-5 (concurrent
        independent faults in different DP groups).
        """
        all_events: list[TelemetryEvent] = []
        for i, spec in enumerate(burst.faults):
            if i > 0 and burst.inter_fault_delay_secs > 0:
                await asyncio.sleep(burst.inter_fault_delay_secs)
            events = await self.inject(spec)
            all_events.extend(events)
        return all_events

    async def inject_rate(
        self,
        spec: FaultSpec,
        rate_per_sec: float,
        duration_secs: float,
    ) -> list[TelemetryEvent]:
        """
        Inject faults at a sustained rate for a fixed duration.

        Used for UT-A1: "inject 10k synthetic events/s, no event loss."
        """
        interval = 1.0 / rate_per_sec
        deadline = time.monotonic() + duration_secs
        all_events: list[TelemetryEvent] = []

        while time.monotonic() < deadline:
            events = await self.inject(
                FaultSpec(
                    fault_signal=spec.fault_signal,
                    rank=spec.rank,
                    node=spec.node,
                    nic_id=spec.nic_id,
                    rack_id=spec.rack_id,
                )
            )
            all_events.extend(events)
            await asyncio.sleep(interval)

        return all_events

    # ------------------------------------------------------------------
    # Observability

    @property
    def injected_count(self) -> int:
        """Total number of events published by this harness."""
        return len(self._injected)

    @property
    def injected_events(self) -> list[TelemetryEvent]:
        """All events published, in order."""
        return list(self._injected)
