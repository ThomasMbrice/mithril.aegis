"""
IT-5 — Simultaneous independent B0 + B1 in different DP groups.

Scenario: NIC flap in DP group A and a node crash in DP group B happen at
the same time (different nodes, different racks).
Pass criterion (§5.2):
  - Both faults handled at their correct tier (B0 and B1 respectively)
  - No interference between them
  - No double-escalation (each fault escalated at most once)
  - Epoch counter has no collisions (each fault gets its own epoch)
"""

from __future__ import annotations

import asyncio

from aegis.runtime import AegisRuntime
from aegis.telemetry.events import FaultSignal, TelemetryEvent


async def _settle(rt: AegisRuntime, expected: int, timeout: float = 0.5) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while len(rt.epe.history) < expected:
        if asyncio.get_event_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)


async def test_concurrent_b0_b1_different_nodes(runtime: AegisRuntime) -> None:
    """
    Publish B0 (node0/rack0) and B1 (node2/rack1) concurrently.
    Both must be handled at their own tier with no cross-contamination.
    """
    await asyncio.gather(
        runtime.utp.publish(TelemetryEvent(
            rank=0, node="node0", fault_signal=FaultSignal.LINK_FLUCTUATION,
            raw_payload={}, epoch=0, nic_id="nic0", rack_id="rack0",
        )),
        runtime.utp.publish(TelemetryEvent(
            rank=2, node="node2", fault_signal=FaultSignal.NODE_CRASH,
            raw_payload={}, epoch=0, rack_id="rack1",
        )),
    )
    await _settle(runtime, expected=2)

    audit = runtime.epe.escalation_audit()
    assert len(audit) == 2

    by_node = {r["node"]: r for r in audit}
    assert by_node["node0"]["final_tier"] == "B0"
    assert by_node["node2"]["final_tier"] == "B1"
    assert all(r["success"] for r in audit)


async def test_no_epoch_collision_under_concurrent_faults(runtime: AegisRuntime) -> None:
    """Concurrent faults must receive unique epochs (UT-A3 property in context)."""
    N = 10
    events = [
        TelemetryEvent(
            rank=i % 4,
            node=f"node{i % 4}",
            fault_signal=FaultSignal.NIC_PORT_FLAP if i % 2 == 0 else FaultSignal.NODE_CRASH,
            raw_payload={},
            epoch=0,
            nic_id="nic0" if i % 2 == 0 else None,
        )
        for i in range(N)
    ]
    await asyncio.gather(*(runtime.utp.publish(e) for e in events))
    await _settle(runtime, expected=N)

    epochs = [r.epoch for r in runtime.epe.history]
    assert len(set(epochs)) == N, (
        f"Epoch collision: {N} faults but only {len(set(epochs))} unique epochs"
    )


async def test_b0_and_b1_do_not_interfere_with_each_other(runtime: AegisRuntime) -> None:
    """
    Faults on node0 (B0) and node1 (B1) should not affect each other's
    tier assignments or escalation state.
    """
    await runtime.utp.publish(TelemetryEvent(
        rank=0, node="node0", fault_signal=FaultSignal.NIC_PORT_FLAP,
        raw_payload={}, epoch=0, nic_id="nic0",
    ))
    await runtime.utp.publish(TelemetryEvent(
        rank=1, node="node1", fault_signal=FaultSignal.OOM_KILLED_RANK,
        raw_payload={}, epoch=0,
    ))
    await _settle(runtime, expected=2)

    audit = runtime.epe.escalation_audit()
    node0_record = next(r for r in audit if r["node"] == "node0")
    node1_record = next(r for r in audit if r["node"] == "node1")

    # B0 fault on node0 must not have triggered B1 escalation
    assert node0_record["final_tier"] == "B0"
    # B1 fault on node1 must not have been pulled up by node0's B0 classification
    assert node1_record["final_tier"] == "B1"
    assert not node0_record["escalated"]
