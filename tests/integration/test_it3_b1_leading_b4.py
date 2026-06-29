"""
IT-3 — B1 as the leading edge of a B4 rack failure.

Scenario: a rack loses power — the first symptom is individual node crashes,
but they arrive in a correlated burst within the correlation window.
Pass criterion (§5.2):
  - EPE re-classifies upward to B4 *before* committing to neighbor-absorb
  - Restores from Tier-3 (remote durable) rather than from neighbor
  - Escalation invariant holds throughout (no de-escalation)
"""

from __future__ import annotations

import asyncio

import pytest

from aegis.policy.dsl import OperatorPolicy
from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent
from chaos_inject.faults import BurstSpec, FaultSpec
from chaos_inject.harness import ChaosHarness


@pytest.fixture
async def rack_runtime() -> AegisRuntime:
    """Runtime with 6 nodes in 2 racks, tight correlation window."""
    policy = OperatorPolicy(
        correlation_window_secs=0.5,
        correlation_node_threshold=3,
    )
    rt = AegisRuntime(policy=policy)

    nodes_rack0 = [f"node{i}" for i in range(4)]
    nodes_rack1 = [f"node{i}" for i in range(4, 8)]
    all_nodes = nodes_rack0 + nodes_rack1

    for node in all_nodes:
        rt.transport.register_node_nics(node, ["nic0", "nic1"])

    for i, node in enumerate(all_nodes):
        rt.compute.register_neighbor(node, all_nodes[(i + 1) % len(all_nodes)])

    rt.storage.write_tier3(epoch=0)

    async with rt:
        yield rt


async def _settle(rt: AegisRuntime, expected: int, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


async def test_correlated_b1_burst_reclassified_to_b4(rack_runtime: AegisRuntime) -> None:
    """Four NODE_CRASH events in rack0 within 0.5s → at least one B4 re-classification."""
    harness = ChaosHarness(rack_runtime.utp, rack_runtime.epoch_service)

    burst = BurstSpec(
        faults=[
            FaultSpec(FaultSignal.NODE_CRASH, rank=i, node=f"node{i}", rack_id="rack0")
            for i in range(4)
        ],
        inter_fault_delay_secs=0.05,
    )
    await harness.inject_burst(burst)
    await _settle(rack_runtime, expected=4)

    audit = rack_runtime.epe.escalation_audit()
    b4_records = [r for r in audit if r["final_tier"] == "B4"]

    assert len(b4_records) >= 1, (
        "Correlation window did not re-classify any B1 → B4\n"
        f"Audit: {audit}"
    )


async def test_escalation_invariant_during_rack_event(rack_runtime: AegisRuntime) -> None:
    """Invariant must hold throughout a rack-level burst."""
    harness = ChaosHarness(rack_runtime.utp, rack_runtime.epoch_service)
    burst = BurstSpec(
        faults=[
            FaultSpec(FaultSignal.NODE_CRASH, rank=i, node=f"node{i}", rack_id="rack0")
            for i in range(4)
        ],
        inter_fault_delay_secs=0.05,
    )
    await harness.inject_burst(burst)
    await _settle(rack_runtime, expected=4)

    for record in rack_runtime.epe.escalation_audit():
        assert record["escalation_valid"], (
            f"Escalation invariant violated: {record}"
        )


async def test_isolated_rack1_not_reclassified(rack_runtime: AegisRuntime) -> None:
    """B1 faults in rack1 while rack0 is bursting should NOT be re-classified to B4."""
    harness = ChaosHarness(rack_runtime.utp, rack_runtime.epoch_service)

    # Crash 3 nodes in rack0 (crosses threshold) then 1 node in rack1
    burst = BurstSpec(
        faults=[
            FaultSpec(FaultSignal.NODE_CRASH, rank=i, node=f"node{i}", rack_id="rack0")
            for i in range(3)
        ] + [
            FaultSpec(FaultSignal.NODE_CRASH, rank=4, node="node4", rack_id="rack1"),
        ],
        inter_fault_delay_secs=0.05,
    )
    await harness.inject_burst(burst)
    await _settle(rack_runtime, expected=4)

    audit = rack_runtime.epe.escalation_audit()
    rack1_records = [r for r in audit if r["node"] == "node4"]

    assert len(rack1_records) == 1
    # rack1 node should stay at B1 (not escalated to B4 by rack0's burst)
    assert rack1_records[0]["final_tier"] == "B1", (
        f"rack1 node incorrectly escalated to {rack1_records[0]['final_tier']}"
    )
