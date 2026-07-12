"""
RankHeartbeatSensor — the real B1 sensor for the real-cluster harness.

This is what CLAUDE.md's Phase 1 status table means by "real sensors wired
to ... heartbeat timeouts": unlike ``aegis.telemetry.sensors.SyntheticSensor``
(which only exists to let tests/chaos_inject hand-write events onto the UTP),
this sensor watches a real, live signal — another rank's training-loop
heartbeat file on disk — and publishes a genuine ``TelemetryEvent`` when that
signal goes stale.

Design: local watcher, not cross-rank RPC. Every rank's own training process
runs one instance of this sensor. Each rank refreshes its own heartbeat file
every training step (``realbench.training.step_log.HeartbeatWriter``, driven
from ``w1_llama7b.py``'s step loop). This sensor polls *all* peers' heartbeat
files and reports a peer as dead if its file stops advancing — a SIGKILLed
rank obviously can't detect its own death, so death is always detected by a
survivor, which is the realistic shape of the problem on real hardware.

Why file-polling and not an NCCL-collective-timeout hook: NCCL doesn't expose
a clean async "this collective is hanging" callback without patching the
collective call sites themselves (a "real NCCL shim" — explicitly called out
in CLAUDE.md as still hardware-pending, out of scope here). Heartbeat-file
polling requires no NCCL internals and is enough to trigger a real B1 UTP
event when ``chaos_inject.real_injector`` genuinely SIGKILLs a rank.

Reuses ``FaultSignal.NODE_CRASH`` -> classified B1 by the existing
``FailureClassifier`` table unchanged (aegis/classifier/classifier.py) —
zero changes needed to the classifier or EPE.

Note on clocks: ``TelemetryEvent.timestamp`` defaults to ``time.monotonic()``
(see aegis/telemetry/events.py) and is not comparable to the wall-clock
timestamps in the step-log / chaos-inject log that ``realbench.collector.align``
aligns against (test_suite.md §4.5.4 requires "same clock"). This sensor
therefore also appends a wall-clock-stamped record to a separate detection
log (``detection_log.jsonl``) at the moment it publishes each event, so the
alignment script has a wall-clock "time-to-detect" signal without needing to
change the shared ``TelemetryEvent`` schema.
"""

from __future__ import annotations

import asyncio
import logging
import time

from aegis.epoch.service import FaultEpochService
from aegis.telemetry.bus import UnifiedTelemetryPlane
from aegis.telemetry.events import FaultSignal, TelemetryEvent
from aegis.telemetry.sensors import SensorBase

from realbench.training.step_log import HeartbeatReader, append_jsonl

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_TIMEOUT_SECS = 5.0
DEFAULT_POLL_INTERVAL_SECS = 1.0


class RankHeartbeatSensor(SensorBase):
    """
    Polls peer heartbeat files; publishes a B1 ``TelemetryEvent`` for any
    peer whose heartbeat has gone stale for longer than ``heartbeat_timeout_secs``.

    ``own_rank`` is excluded from polling (a rank cannot detect its own death).
    ``peer_heartbeat_paths`` maps rank -> path to that rank's heartbeat file.
    """

    def __init__(
        self,
        utp: UnifiedTelemetryPlane,
        epoch_service: FaultEpochService,
        own_rank: int,
        peer_heartbeat_paths: dict[int, str],
        detection_log_path: str,
        heartbeat_timeout_secs: float = DEFAULT_HEARTBEAT_TIMEOUT_SECS,
        poll_interval_secs: float = DEFAULT_POLL_INTERVAL_SECS,
    ) -> None:
        super().__init__(utp, source="rank_heartbeat")
        self._epochs = epoch_service
        self._own_rank = own_rank
        self._peers = {
            rank: HeartbeatReader(path)
            for rank, path in peer_heartbeat_paths.items()
            if rank != own_rank
        }
        self._detection_log_path = detection_log_path
        self._timeout = heartbeat_timeout_secs
        self._interval = poll_interval_secs
        self._already_reported: set[int] = set()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="rank-heartbeat-sensor")

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        while self._running:
            now = time.time()
            for rank, reader in self._peers.items():
                record = reader.read()
                if record is None:
                    # No heartbeat written yet (peer still starting up) — not a fault.
                    continue
                stale = (now - record.wall_clock) > self._timeout
                if stale and rank not in self._already_reported:
                    await self._report_dead(rank)
                elif not stale and rank in self._already_reported:
                    # Heartbeat resumed — clear so a future death fires again.
                    self._already_reported.discard(rank)
            await asyncio.sleep(self._interval)

    async def _report_dead(self, dead_rank: int) -> None:
        self._already_reported.add(dead_rank)
        detect_wall_clock = time.time()
        event = TelemetryEvent(
            rank=dead_rank,
            node=f"rank{dead_rank}",
            fault_signal=FaultSignal.NODE_CRASH,
            raw_payload={"detected_by_rank": self._own_rank},
            epoch=self._epochs.current(),
            source="rank_heartbeat",
        )
        logger.warning(
            "[rank_heartbeat] rank %d detected peer rank %d dead (no heartbeat for >%.1fs)",
            self._own_rank, dead_rank, self._timeout,
        )
        append_jsonl(
            self._detection_log_path,
            {
                "wall_clock": detect_wall_clock,
                "dead_rank": dead_rank,
                "detected_by_rank": self._own_rank,
                "epoch": event.epoch,
            },
        )
        await self.emit(event)
