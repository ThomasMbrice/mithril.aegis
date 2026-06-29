"""
Layer B — Transport survivability (B0).

Stub for R²CCL (arXiv:2512.25059).  In production this wraps:
  - NCCL interception shim (sit on the collective path)
  - QP migration to backup NIC (primary-backup model)
  - R2CC-Balance: bandwidth-aware AllReduce redistribution
  - Fast-path autonomy: act in ms–s, publish to UTP concurrently

Reported numbers: <1% training / <3% inference overhead;
85–89% collective throughput retained under single-NIC failure.

Prerequisite (§2.2): ≥2 RNICs per node.  Nodes with only one NIC cannot
absorb B0 faults at this tier — they fall through to B1+.
"""

from __future__ import annotations

import asyncio
import logging

from .base import RecoveryLayer, RecoveryResult
from ..telemetry.events import BlastRadius, TelemetryEvent

logger = logging.getLogger(__name__)

_B0_TIERS = frozenset({BlastRadius.B0})


class TransportLayer(RecoveryLayer):
    """
    B0 recovery via NIC connection migration.

    Fast-path autonomy (§3.1): R²CCL may begin migration immediately and
    publish to UTP concurrently.  The classifier only intervenes if the
    fault escalates beyond B0.
    """

    def __init__(self) -> None:
        # node → list of available NIC identifiers
        self._available_nics: dict[str, list[str]] = {}

    @property
    def handled_tiers(self) -> frozenset[BlastRadius]:
        return _B0_TIERS

    async def can_handle(self, event: TelemetryEvent, tier: BlastRadius) -> bool:
        if tier != BlastRadius.B0:
            return False
        # Multi-NIC prerequisite: node must have ≥2 NICs registered
        backups = self._available_nics.get(event.node, [])
        return len(backups) >= 2  # noqa: PLR2004

    async def recover(
        self, event: TelemetryEvent, tier: BlastRadius, epoch: int
    ) -> RecoveryResult:
        """
        Stub: simulate QP migration to backup NIC.

        Real: NCCL shim intercepts, migrates queue pair to backup RNIC,
        restores collective with R2CC-Balance bandwidth redistribution.
        Target overhead: <1% training, <3% inference.
        """
        nic = event.nic_id or "primary"
        logger.info(
            "[B0] Transport recovery: migrating NIC %s → backup on node %s (epoch %d)",
            nic,
            event.node,
            epoch,
        )
        await asyncio.sleep(0)  # real: QP migration + bandwidth rebalance

        return RecoveryResult(
            success=True,
            message=f"NIC migration complete: {nic} → backup on {event.node}",
            degraded=False,
        )

    # ------------------------------------------------------------------
    # Configuration

    def register_node_nics(self, node: str, nics: list[str]) -> None:
        """
        Register the NICs available on a node.

        Must be called before the runtime processes any B0 faults for that
        node.  The first NIC in the list is the primary; subsequent entries
        are backup candidates.
        """
        self._available_nics[node] = list(nics)
