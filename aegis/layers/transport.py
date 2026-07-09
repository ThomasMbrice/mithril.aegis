"""
Layer B — Transport survivability (B0).

Real implementation for R²CCL (arXiv:2512.25059)'s two software-only pieces:
  - Connection migration state machine (primary-backup QP model)
  - R2CC-Balance: bandwidth-aware redistribution across surviving NICs

Both are genuine, deterministic, unit-testable logic — no NCCL/RDMA
hardware required to exercise them.  What's *not* implemented here is the
NCCL interception shim itself and the actual QP/RDMA migration syscalls,
since those need real multi-NIC IB/RoCE hardware (§2.2) to validate.  That
boundary is expressed as a pluggable ``TransportBackend``: the default
``SimulatedTransportBackend`` does real bookkeeping and real bandwidth
math against synthetic capacities; ``LinuxRDMABackend`` is the
hardware-pending path meant for the A100/IB cluster and is untested on
this development machine (no RNICs here).

Reported paper numbers (<1% training / <3% inference overhead; 85-89%
collective throughput retained) describe measurements on the paper's own
IB testbed and are NOT reproduced by ``SimulatedTransportBackend`` — this
layer computes a real capacity-based throughput-retained figure from
whatever NIC bandwidths are registered, which is an honest software proxy,
not a hardware validation of the paper's claim (see design.md §8.1).

Prerequisite (§2.2): ≥2 RNICs per node.  Nodes with only one NIC cannot
absorb B0 faults at this tier — they fall through to B1+.
"""

from __future__ import annotations

import abc
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from .base import RecoveryLayer, RecoveryResult
from ..telemetry.events import BlastRadius, TelemetryEvent

logger = logging.getLogger(__name__)

_B0_TIERS = frozenset({BlastRadius.B0})

# Default per-NIC bandwidth (Gbps) when a caller doesn't specify capacities —
# symmetric NICs is the common case and matches the paper's testbed shape.
_DEFAULT_NIC_BANDWIDTH_GBPS = 100.0


class NicState(str, Enum):
    """State machine for a node's transport connection."""

    STABLE = "stable"
    MIGRATING = "migrating"
    MIGRATED = "migrated"
    FAILED = "failed"


@dataclass
class NodeNics:
    """Per-node NIC registry: identifiers, capacities, and live state."""

    nics: list[str]
    bandwidth_gbps: dict[str, float] = field(default_factory=dict)
    state: NicState = NicState.STABLE
    active_nic: str = ""
    failed_nics: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.active_nic and self.nics:
            self.active_nic = self.nics[0]
        for nic in self.nics:
            self.bandwidth_gbps.setdefault(nic, _DEFAULT_NIC_BANDWIDTH_GBPS)

    @property
    def healthy_nics(self) -> list[str]:
        return [n for n in self.nics if n not in self.failed_nics]


@dataclass
class MigrationResult:
    """Result of a real (simulated-hardware or real-hardware) NIC migration."""

    success: bool
    from_nic: str
    to_nic: str
    migration_secs: float
    throughput_retained_pct: float
    rebalanced_weights: dict[str, float] = field(default_factory=dict)


def rebalance_bandwidth(remaining: dict[str, float]) -> dict[str, float]:
    """
    R2CC-Balance: redistribute AllReduce traffic across surviving NICs,
    weighted by each NIC's own capacity.

    Real, deterministic math — not a hardware measurement.  Given the
    remaining healthy NICs' capacities, returns the fraction of traffic
    each should now carry (sums to 1.0).
    """
    total = sum(remaining.values())
    if total <= 0:
        return {nic: 0.0 for nic in remaining}
    return {nic: cap / total for nic, cap in remaining.items()}


class TransportBackend(abc.ABC):
    """Pluggable mechanism that actually performs the NIC migration."""

    @abc.abstractmethod
    async def migrate(
        self,
        node: str,
        from_nic: str,
        to_nic: str,
        total_capacity_before: float,
        remaining_capacity: dict[str, float],
    ) -> MigrationResult:
        """Migrate the connection from ``from_nic`` to ``to_nic`` on ``node``."""


class SimulatedTransportBackend(TransportBackend):
    """
    Default backend.  Real state-machine + real R2CC-Balance bandwidth math,
    no OS/hardware calls.  Charges a small, real, measured wall-clock cost
    for the migration itself so repeated recovery-time measurements are
    non-zero and comparable, rather than an instantaneous no-op.
    """

    # Sub-millisecond QP-swap overhead is what R²CCL targets; we charge a
    # deliberately small but nonzero cost so timing metrics are meaningful.
    MIGRATION_OVERHEAD_SECS = 0.002

    async def migrate(
        self,
        node: str,
        from_nic: str,
        to_nic: str,
        total_capacity_before: float,
        remaining_capacity: dict[str, float],
    ) -> MigrationResult:
        start = time.perf_counter()
        await asyncio.sleep(self.MIGRATION_OVERHEAD_SECS)
        elapsed = time.perf_counter() - start

        total_after = sum(remaining_capacity.values())
        retained_pct = (
            (total_after / total_capacity_before) * 100.0
            if total_capacity_before > 0
            else 0.0
        )
        weights = rebalance_bandwidth(remaining_capacity)

        return MigrationResult(
            success=True,
            from_nic=from_nic,
            to_nic=to_nic,
            migration_secs=elapsed,
            throughput_retained_pct=retained_pct,
            rebalanced_weights=weights,
        )


class LinuxRDMABackend(TransportBackend):
    """
    Hardware-pending path for the real A100/IB cluster (§2.2: NCCL,
    multi-NIC hardware, RDMA/InfiniBand or RoCE).  Queries the backup
    RNIC's link state via ``ip link`` before declaring migration complete.

    Requires root, Linux, and ≥2 real RNICs — none of which this
    development machine has.  This path has not been exercised against
    real hardware; validate on the A100/IB cluster before trusting it
    (§6 risk: NCCL ABI drift; multi-NIC hardware prerequisite).
    """

    async def migrate(
        self,
        node: str,
        from_nic: str,
        to_nic: str,
        total_capacity_before: float,
        remaining_capacity: dict[str, float],
    ) -> MigrationResult:
        start = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                "ip", "link", "show", to_nic,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            up = proc.returncode == 0
        except FileNotFoundError:
            logger.warning(
                "[B0] LinuxRDMABackend: `ip` not available on this host — "
                "cannot validate backup NIC %s; this backend requires the "
                "real A100/IB cluster",
                to_nic,
            )
            up = False
        elapsed = time.perf_counter() - start

        total_after = sum(remaining_capacity.values())
        retained_pct = (
            (total_after / total_capacity_before) * 100.0
            if total_capacity_before > 0
            else 0.0
        )
        return MigrationResult(
            success=up,
            from_nic=from_nic,
            to_nic=to_nic,
            migration_secs=elapsed,
            throughput_retained_pct=retained_pct if up else 0.0,
            rebalanced_weights=rebalance_bandwidth(remaining_capacity) if up else {},
        )


class TransportLayer(RecoveryLayer):
    """
    B0 recovery via real NIC connection-migration state machine + real
    R2CC-Balance bandwidth math, on a pluggable backend.

    Fast-path autonomy (§3.1): R²CCL may begin migration immediately and
    publish to UTP concurrently.  The classifier only intervenes if the
    fault escalates beyond B0.
    """

    def __init__(self, backend: TransportBackend | None = None) -> None:
        self._nodes: dict[str, NodeNics] = {}
        self._backend: TransportBackend = backend or SimulatedTransportBackend()
        self.last_migration: MigrationResult | None = None

    @property
    def handled_tiers(self) -> frozenset[BlastRadius]:
        return _B0_TIERS

    async def can_handle(self, event: TelemetryEvent, tier: BlastRadius) -> bool:
        if tier != BlastRadius.B0:
            return False
        entry = self._nodes.get(event.node)
        if entry is None:
            return False
        # Multi-NIC prerequisite: node must have ≥2 healthy NICs
        return len(entry.healthy_nics) >= 2  # noqa: PLR2004

    async def recover(
        self,
        event: TelemetryEvent,
        tier: BlastRadius,
        epoch: int,
        *,
        min_valid_epoch: int | None = None,
    ) -> RecoveryResult:
        entry = self._nodes.get(event.node)
        if entry is None:
            return RecoveryResult(success=False, message=f"No NICs registered for {event.node}")

        failed_nic = event.nic_id or entry.active_nic
        if failed_nic not in entry.nics:
            failed_nic = entry.active_nic

        candidates = [n for n in entry.healthy_nics if n != failed_nic]
        if not candidates:
            entry.state = NicState.FAILED
            return RecoveryResult(
                success=False,
                message=f"No backup NIC available on {event.node} for {failed_nic}",
            )

        backup_nic = candidates[0]
        entry.state = NicState.MIGRATING
        entry.failed_nics.add(failed_nic)

        total_before = sum(entry.bandwidth_gbps.values())
        remaining = {n: entry.bandwidth_gbps[n] for n in entry.healthy_nics}

        result = await self._backend.migrate(
            event.node, failed_nic, backup_nic, total_before, remaining
        )
        self.last_migration = result

        if not result.success:
            entry.state = NicState.FAILED
            return RecoveryResult(
                success=False,
                message=f"NIC migration failed: {failed_nic} → {backup_nic} on {event.node}",
            )

        entry.active_nic = backup_nic
        entry.state = NicState.MIGRATED

        logger.info(
            "[B0] Transport recovery: %s → %s on %s (epoch %d, %.1fus, "
            "%.1f%% throughput retained)",
            failed_nic, backup_nic, event.node, epoch,
            result.migration_secs * 1e6, result.throughput_retained_pct,
        )

        return RecoveryResult(
            success=True,
            message=(
                f"NIC migration complete: {failed_nic} → {backup_nic} on {event.node} "
                f"({result.throughput_retained_pct:.1f}% throughput retained)"
            ),
            degraded=False,
        )

    # ------------------------------------------------------------------
    # Configuration

    def register_node_nics(
        self,
        node: str,
        nics: list[str],
        bandwidth_gbps: dict[str, float] | None = None,
    ) -> None:
        """
        Register the NICs available on a node.

        Must be called before the runtime processes any B0 faults for that
        node.  The first NIC in the list is the primary; subsequent entries
        are backup candidates.  ``bandwidth_gbps`` optionally overrides the
        default symmetric-capacity assumption per NIC — used by
        ``rebalance_bandwidth`` for the throughput-retained calculation.
        """
        self._nodes[node] = NodeNics(nics=list(nics), bandwidth_gbps=dict(bandwidth_gbps or {}))

    def node_state(self, node: str) -> NicState | None:
        entry = self._nodes.get(node)
        return entry.state if entry else None
