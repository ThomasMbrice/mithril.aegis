"""
UT-B — Real TransportLayer (B0) math and state machine.

Not a reproduction of R²CCL's <1%/<3% overhead or 85-89% throughput
numbers (those need real IB hardware, see design.md §8.1) — these tests
validate that the *software* pieces (NIC state machine, R2CC-Balance
bandwidth redistribution math) are real and correct.
"""

from __future__ import annotations

import pytest

from aegis.layers.transport import (
    NicState,
    SimulatedTransportBackend,
    TransportLayer,
    rebalance_bandwidth,
)
from aegis.telemetry.events import BlastRadius, FaultSignal, TelemetryEvent


def _event(node: str, nic_id: str, rank: int = 0) -> TelemetryEvent:
    return TelemetryEvent(
        rank=rank, node=node, fault_signal=FaultSignal.NIC_PORT_FLAP,
        raw_payload={}, epoch=0, nic_id=nic_id,
    )


def test_rebalance_bandwidth_proportional_to_capacity():
    """Real R2CC-Balance math: surviving NICs' share is proportional to their own capacity."""
    weights = rebalance_bandwidth({"nic1": 100.0, "nic2": 200.0})
    assert weights["nic1"] == pytest.approx(1 / 3)
    assert weights["nic2"] == pytest.approx(2 / 3)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_rebalance_bandwidth_empty_remaining():
    assert rebalance_bandwidth({}) == {}


async def test_migration_retains_capacity_ratio_symmetric_nics():
    """Losing 1 of 2 equal-capacity NICs retains exactly half the aggregate capacity."""
    layer = TransportLayer()
    layer.register_node_nics("node0", ["nic0", "nic1"], bandwidth_gbps={"nic0": 100.0, "nic1": 100.0})

    result = await layer.recover(_event("node0", "nic0"), BlastRadius.B0, epoch=1)

    assert result.success
    assert not result.degraded
    assert layer.last_migration is not None
    assert layer.last_migration.throughput_retained_pct == pytest.approx(50.0)
    assert layer.node_state("node0") == NicState.MIGRATED


async def test_migration_retains_weighted_capacity_asymmetric_nics():
    """A bigger backup NIC retains more than half of aggregate capacity."""
    layer = TransportLayer()
    layer.register_node_nics(
        "node0", ["nic0", "nic1"], bandwidth_gbps={"nic0": 100.0, "nic1": 300.0}
    )

    result = await layer.recover(_event("node0", "nic0"), BlastRadius.B0, epoch=1)

    assert result.success
    assert layer.last_migration.throughput_retained_pct == pytest.approx(75.0)


async def test_can_handle_requires_two_healthy_nics():
    layer = TransportLayer()
    layer.register_node_nics("solo", ["nic0"])
    assert not await layer.can_handle(_event("solo", "nic0"), BlastRadius.B0)

    layer.register_node_nics("duo", ["nic0", "nic1"])
    assert await layer.can_handle(_event("duo", "nic0"), BlastRadius.B0)


async def test_after_migration_only_one_healthy_nic_remains():
    """A second failure on the same node (now down to 1 NIC) can no longer be absorbed at B0."""
    layer = TransportLayer()
    layer.register_node_nics("node0", ["nic0", "nic1"])
    await layer.recover(_event("node0", "nic0"), BlastRadius.B0, epoch=1)

    assert not await layer.can_handle(_event("node0", "nic1"), BlastRadius.B0)


async def test_unregistered_node_cannot_recover():
    layer = TransportLayer()
    result = await layer.recover(_event("ghost", "nic0"), BlastRadius.B0, epoch=1)
    assert not result.success


async def test_simulated_backend_migration_is_real_measured_time():
    """The simulated backend charges a real, nonzero, measured wall-clock cost."""
    backend = SimulatedTransportBackend()
    result = await backend.migrate("node0", "nic0", "nic1", 200.0, {"nic1": 100.0})
    assert result.success
    assert result.migration_secs > 0.0
    assert result.migration_secs < 1.0  # sub-second QP swap, not a full restart
